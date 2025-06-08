import queue
import time
from collections import deque
from typing import Dict, List, Optional, Callable, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from repairs_components.processing.scene_creation_funnel import (
    create_env_configs,
)
from repairs_components.processing.tasks import Task
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.training_utils.sim_state_global import merge_global_states
from repairs_components.training_utils.concurrent_scene_dataclass import (
    ConcurrentSceneData,
    merge_concurrent_scene_configs,
)
import genesis as gs
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.processing.tasks import Task


class MultiEnvDataLoader:
    """
    Multi-environment DataLoader supporting asynchronous prefetching of environment configurations.

    Call `populate_async(request_tensor)` to enqueue a specified number of configs per environment,
    and `get_processed_data(request_tensor)` to retrieve processed data and auto-top-up queues.
    """

    def __init__(
        self,
        num_environments: int,  # in general called "environments" though it's scenes.
        preprocessing_fn: Callable,
        prefetch_memory_size: int = 256,
        max_workers: int = 4,
    ):
        """
        Multi-environment dataloader with selective prefetching.

        Args:
            num_environments: Total number of environments
            preprocessing_fn: Function that takes (env_idx, raw_data) -> processed_data
            prefetch_size: How many items to prefetch per environment
            max_workers: Number of worker threads for preprocessing
        """
        self.num_environments = num_environments
        self.preprocessing_fn = preprocessing_fn
        self.prefetch_size = prefetch_memory_size
        self.max_workers = max_workers

        # Per-environment queues for prefetched data
        self.prefetch_queues: Dict[int, queue.Queue] = {
            i: queue.Queue(maxsize=prefetch_memory_size)
            for i in range(num_environments)
        }

        # Threading components
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def get_processed_data(
        self, num_configs_to_generate_per_scene: torch.Tensor, timeout: float = 1.0
    ) -> List[Any]:
        """
        Get preprocessed data for all active environments.

        Args:
            num_configs_to_generate_per_scene: Tensor containing the number of configurations to generate per scene.
                                              Should have length equal to the number of active environments.
            timeout: Max time to wait for data for each scene

        Returns:
            List of preprocessed data for each active environment (len=num_configs_to_generate_per_scene.shape[0])
        """
        assert len(num_configs_to_generate_per_scene) == self.num_environments, (
            f"num_configs_to_generate_per_scene length ({len(num_configs_to_generate_per_scene)}) "
            f"mismatches count of registered envs ({self.num_environments})."
        )
        assert num_configs_to_generate_per_scene.dtype == torch.uint16, (
            "Expected num_configs_to_generate_per_scene to be a uint16 tensor."
        )  # non-negative integer in general.

        num_configs_to_generate = torch.zeros_like(num_configs_to_generate_per_scene)
        results_from_queue = []
        for scene_id, num_to_generate_tensor in enumerate(
            num_configs_to_generate_per_scene
        ):
            num_to_generate = int(num_to_generate_tensor)
            results_this_scene = []
            # FIXME: why is qsize 0 in main thread at init?
            queue_size = self.prefetch_queues[scene_id].qsize()

            # find minimal between num_to_generate and queue_size
            take = min(
                num_to_generate, queue_size
            )  # it won't happen async anyway, I understand.
            for _ in range(take):
                results_this_scene.append(
                    self.prefetch_queues[scene_id].get(timeout=timeout)
                )
            num_configs_to_generate[scene_id] = num_to_generate - take
            results_from_queue.append(results_this_scene)

        # generator - count however much is not enough and gen it.
        starved_configs = self.preprocessing_fn(num_configs_to_generate)

        # log issues is starvation is over 30%.
        total_insufficient_count = torch.sum(num_configs_to_generate)
        total_requested_count = torch.sum(num_configs_to_generate_per_scene)
        if total_insufficient_count > total_requested_count * 0.3:
            print(
                "Warning: experiencing environment count starvation. Insufficient_gen: "
                + str(total_insufficient_count.item())
                + " of "
                + str(total_requested_count.item())
            )

        # put newly generated configs to queue (to alleviate starvation)
        # note: in future, configs should be reused 3-4 times before being discarded.
        for scene_id, num_to_generate_tensor in enumerate(num_configs_to_generate):
            num_to_generate = int(num_to_generate_tensor.item())
            for _ in range(num_to_generate):
                self.prefetch_queues[scene_id].put(starved_configs[scene_id])

        # Merge the queue configs and the starved configs
        for i in range(len(num_configs_to_generate_per_scene)):
            # currently the result is list of lists
            total_configs_this_queue = results_from_queue[i] + starved_configs[i]
            # process the result to be a single ConcurrentState
            results_from_queue[i] = merge_concurrent_scene_configs(
                total_configs_this_queue
            )

        self.populate_async(num_configs_to_generate_per_scene)
        return results_from_queue

    def populate_async(self, num_configs_to_generate_per_scene: torch.Tensor) -> Any:
        """
        Asynchronously populate prefetch queues with specified number of configs per environment.
        Args:
            num_configs_to_generate_per_scene: Tensor with number of configs per environment.
        Returns:
            Future object for the population task.
        """
        assert num_configs_to_generate_per_scene.shape[0] == self.num_environments, (
            "Expected tensor length equal to num_environments."
        )
        assert num_configs_to_generate_per_scene.dtype == torch.uint16, (
            "Expected uint16 tensor for num_configs_to_generate_per_scene."
        )
        future = self.executor.submit(
            self._populate_worker, num_configs_to_generate_per_scene
        )
        return future

    def _populate_worker(self, num_configs_to_generate: torch.Tensor):
        """
        Worker that generates data and enqueues it per environment asynchronously.
        """
        batch = self.preprocessing_fn(num_configs_to_generate)
        for idx, items in enumerate(batch):
            q = self.prefetch_queues[idx]
            for it in items:
                q.put(it)


class RepairsEnvDataLoader(MultiEnvDataLoader):
    def __init__(
        self,
        scenes: List[gs.Scene],
        env_setups: List[EnvSetup],
        tasks: List[Task],
        batch_dim: int = 128,
        prefetch_memory_size=256,
    ):
        """
        Repairs environment dataloader that prefetches scene data.

        Args:
            scenes: List of scene templates
            env_setups: List of environment setups
            tasks: List of tasks
            batch_dim: Batch dimension size
            num_scenes_per_task: Number of scenes per task
            batches_in_memory_per_scene: How many batches to keep in memory per scene
        """
        self.scenes = scenes
        self.env_setups = env_setups
        self.tasks = tasks
        self.batch_dim = batch_dim

        assert len(scenes) == len(env_setups), (
            "Count of scenes and env_setups must match."
        )

        # Create preprocessing function that generates batches for scenes
        def scene_preprocessing_fn(
            num_configs_to_generate_per_scene: torch.Tensor,
            _state_data: Any | None = None,
        ) -> List[ConcurrentSceneData]:
            return self._generate_scene_batches(num_configs_to_generate_per_scene)

        super().__init__(
            num_environments=len(scenes),
            preprocessing_fn=scene_preprocessing_fn,
            prefetch_memory_size=prefetch_memory_size,
        )

    def _generate_scene_batches(
        self, num_configs_to_generate_per_scene: torch.Tensor
    ) -> List[ConcurrentSceneData]:
        """Generate batches for a specific scene.

        Inputs:
        num_configs_to_generate_per_scene: torch.Tensor of shape [num_tasks_per_scene] with torch_bool."""
        # FIXME: I also need a possibility of returning controlled-size batches.
        # in fact, I don't need batches in memory at all, I need individual items.
        assert len(self.env_setups) == self.num_environments, (
            "One env setup per one env"
        )
        assert len(self.env_setups) == len(num_configs_to_generate_per_scene), (
            "One env setup per config to generate."
        )
        # Generate one "chunk" of data for this scene # even though I don't need to generate a chunk.

        scene_configs_per_scene = create_env_configs(
            env_setups=self.env_setups,
            tasks=self.tasks,
            num_configs_to_generate_per_scene=num_configs_to_generate_per_scene,
        )

        batches = []
        for scene_idx in range(len(num_configs_to_generate_per_scene)):
            batch_starting_states = merge_global_states(
                [scene_configs_per_scene[scene_idx].current_state]
            )
            batch_desired_states = merge_global_states(
                [scene_configs_per_scene[scene_idx].desired_state]
            )
            vox_init = scene_configs_per_scene[scene_idx].vox_init
            vox_des = scene_configs_per_scene[scene_idx].vox_des

            initial_diffs, initial_diff_count = batch_starting_states.diff(
                batch_desired_states
            )

            # Create ConcurrentSceneData object
            batch_data = ConcurrentSceneData(
                scene=None,
                gs_entities=None,
                cameras=None,
                current_state=batch_starting_states,
                desired_state=batch_desired_states,
                vox_init=vox_init,
                vox_des=vox_des,
                initial_diffs=initial_diffs,
                initial_diff_counts=initial_diff_count,
                scene_id=scene_idx,
            )
            batches.append(batch_data)

        return batches
