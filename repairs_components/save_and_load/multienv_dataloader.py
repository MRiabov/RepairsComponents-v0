from __future__ import annotations

import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import genesis as gs
import torch

from repairs_components.processing.scene_creation_funnel import create_env_configs
from repairs_components.processing.tasks import Task
from repairs_components.save_and_load.offline_dataloading import OfflineDataloader
from repairs_components.training_utils.concurrent_scene_dataclass import (
    ConcurrentSceneData,
    merge_concurrent_scene_configs,
    split_scene_config,
)
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.training_utils.offline_utils import (
    load_env_configs_from_disk,
    save_env_configs_to_disk,
)


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

        # impl note: num_configs_to_generate_per_scene is total configs to generate, and
        assert len(num_configs_to_generate_per_scene) == self.num_environments, (
            f"num_configs_to_generate_per_scene length ({len(num_configs_to_generate_per_scene)}) "
            f"mismatches count of registered envs ({self.num_environments})."
        )
        assert num_configs_to_generate_per_scene.dtype == torch.int16, (
            "Expected num_configs_to_generate_per_scene to be a int16 tensor."
        )  # non-negative integer in general.

        count_insufficient_configs_per_scene = torch.zeros_like(
            num_configs_to_generate_per_scene
        )
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
            count_insufficient_configs_per_scene[scene_id] = num_to_generate - take
            results_from_queue.append(results_this_scene)

        # generator - count however much is not enough and gen it.
        if torch.any(count_insufficient_configs_per_scene > 0):
            starved_configs = self.preprocessing_fn(
                count_insufficient_configs_per_scene
            )
        else:
            starved_configs = []

        # log issues is starvation is over 30%.
        total_insufficient_count = torch.sum(count_insufficient_configs_per_scene)
        total_requested_count = torch.sum(num_configs_to_generate_per_scene)
        if total_insufficient_count > total_requested_count * 0.3:
            print(
                "Warning: experiencing environment count starvation. Insufficient_gen: "
                + str(total_insufficient_count.item())
                + " of "
                + str(total_requested_count.item())
            )

        # put newly generated configs to queue (to alleviate starvation)
        # enqueue each starved config individually per scene
        for scene_id, cfg_list in enumerate(starved_configs):
            for cfg in cfg_list:
                try:
                    self.prefetch_queues[scene_id].put_nowait(cfg)
                except queue.Full:
                    pass  # if it's already full, nothing bad happened, skip.

        # Merge the queue configs and the starved configs
        total_configs = []
        for scene_id in range(len(num_configs_to_generate_per_scene)):
            # extend with any starved configs
            if count_insufficient_configs_per_scene[scene_id] > 0:
                results_from_queue[scene_id].extend(starved_configs[scene_id])
            # process the result to be a single ConcurrentState
            total_configs.append(
                merge_concurrent_scene_configs(results_from_queue[scene_id])
            )

        # automatically refill the configs that were taken.
        self.populate_async(
            num_configs_to_generate_per_scene.to(dtype=torch.int16)
        )  # note: mb only ones which were *taken*.

        assert all(
            [
                cfg.initial_diff_counts.shape[0] == num_configs_to_generate_per_scene[i]
                for i, cfg in enumerate(total_configs)
            ]
        ), "Total configs do not match the number of configs to generate."
        return total_configs

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
        assert num_configs_to_generate_per_scene.dtype == torch.int16, (
            "Expected int16 tensor for num_configs_to_generate_per_scene."
        )
        assert (num_configs_to_generate_per_scene >= 0).all(), (
            "num_configs_to_generate_per_scene must be non-negative"
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
                # ensure prefetch items are individual scenes
                try:
                    assert it.current_state.scene_batch_dim == 1, (
                        f"Prefetch item batch_dim={it.current_state.scene_batch_dim}, expected 1"
                    )
                except AssertionError as e:
                    print(f"[Warning] {e}")
                q.put(it)


class RepairsEnvDataLoader(MultiEnvDataLoader):
    def __init__(
        self,
        online: bool,
        *,
        # online
        tasks_to_generate: List[Task] | None = None,
        env_setups: List[EnvSetup] | None = None,
        prefetch_memory_size=256,
        # offline
        env_setup_ids: List[int] | None = None,
        scene_ids: List[int] | None = None,
        offline_data_dir: str | Path | None = "/workspace/data",
    ):
        """
        Repairs environment dataloader that prefetches scene data.

        Args:
            online - whether to load online or offline data.
            env_setup_ids: ONLINE and OFFLINE: List of environment setup ids
            env_setups: ONLINE: List of environment setups to generate from
            scene_ids: OFFLINE: List of scene ids
            tasks_to_generate: ONLINE: List of tasks to generate
            prefetch_memory_size: ONLINE: How many batches to keep in memory (during offline they are all held in memory)
        """
        self.online = online
        self.env_setup_ids = env_setup_ids
        self.tasks_to_generate = tasks_to_generate
        self.offline_dataloader = None

        # Create preprocessing function that generates or loads batches for scenes
        def online_preprocessing_fn(
            num_configs_to_generate_per_scene: torch.Tensor,
        ) -> List[List[ConcurrentSceneData]]:
            return self._generate_scene_batches(num_configs_to_generate_per_scene)

        def get_offline_data_fn(
            num_configs_to_generate_per_scene: torch.Tensor,
        ) -> List[List[ConcurrentSceneData]]:
            return self._load_offline_data(num_configs_to_generate_per_scene)

        if online:
            assert env_setups is not None, (
                "env_setups must be provided for online dataloader"
            )
            assert tasks_to_generate is not None, (
                "tasks_to_generate must be provided for online dataloader"
            )
            self.env_setups = env_setups
            self.tasks = tasks_to_generate
            super().__init__(
                num_environments=len(env_setups),
                preprocessing_fn=online_preprocessing_fn,
                prefetch_memory_size=prefetch_memory_size,
            )
        else:  # offline
            assert env_setup_ids is not None, (
                "env_setup_ids must be provided for offline dataloader"
            )
            assert scene_ids is not None, (
                "scene_ids must be provided for offline dataloader"
            )
            assert offline_data_dir is not None, (
                "data_dir must be provided for offline dataloader"
            )
            # create a cache storage buffer/dataclass
            self.offline_dataloader = OfflineDataloader(
                offline_data_dir,
                env_setup_ids,
            )
            super().__init__(
                num_environments=len(env_setup_ids),
                preprocessing_fn=get_offline_data_fn,
                prefetch_memory_size=prefetch_memory_size,
            )
            # TODO: here - load offline data. Also rename the module back to MultienvDataLoader

    def _generate_scene_batches(
        self, num_configs_to_generate_per_scene: torch.Tensor
    ) -> List[List[ConcurrentSceneData]]:
        """Generate batches of individual configs for a specific scene.

        Inputs:
        num_configs_to_generate_per_scene: torch.Tensor of shape [num_tasks_per_scene] with torch_bool."""
        # FIXME: I also need a possibility of returning controlled-size batches.
        # in fact, I don't need batches in memory at all, I need individual items.

        # note: this method is expected to return individual configs, not batched/merged.
        assert len(self.env_setups) == self.num_environments, (
            "One env setup per one env"
        )
        assert len(self.env_setups) == len(num_configs_to_generate_per_scene), (
            "One env setup per config to generate."
        )
        # Generate one "chunk" of data for this scene # even though I don't need to generate a chunk.

        scene_configs_per_scene, _ = create_env_configs(
            env_setups=self.env_setups,
            tasks=self.tasks,
            num_configs_to_generate_per_scene=num_configs_to_generate_per_scene,
        )

        batches = []
        # Split each batched scene config into individual items (batch dim =1)
        for scene_idx, scene_cfg in enumerate(scene_configs_per_scene):
            assert scene_cfg.batch_dim == num_configs_to_generate_per_scene[scene_idx]
            cfg_list = split_scene_config(scene_cfg)
            batches.append(cfg_list)

        return batches

    def _load_offline_data(self, num_envs_per_scene: torch.Tensor):
        """Load and process batches of individual configs for a specific scene."""
        assert self.offline_dataloader is not None, (
            "Offline dataloader not initialized."
        )
        scene_configs_per_scene = self.offline_dataloader.get_processed_offline_data(
            num_envs_per_scene=num_envs_per_scene,
        )

        batches = []
        # Split each batched scene config into individual items (batch dim =1)
        for scene_idx, scene_cfg in enumerate(scene_configs_per_scene):
            assert scene_cfg.batch_dim == num_envs_per_scene[scene_idx]
            cfg_list = split_scene_config(scene_cfg)
            batches.append(cfg_list)

        return batches
