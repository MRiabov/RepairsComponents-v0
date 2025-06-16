import queue
from typing import Dict, List, Callable, Any
from concurrent.futures import ThreadPoolExecutor

import torch
from repairs_components.processing.scene_creation_funnel import (
    create_env_configs,
)
from repairs_components.processing.tasks import Task
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.training_utils.sim_state_global import (
    RepairsSimState,
)
from repairs_components.training_utils.concurrent_scene_dataclass import (
    ConcurrentSceneData,
    merge_concurrent_scene_configs,
)
import genesis as gs
from repairs_components.training_utils.progressive_reward_calc import RewardHistory


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
        ) -> List[List[ConcurrentSceneData]]:
            return self._generate_scene_batches(num_configs_to_generate_per_scene)

        super().__init__(
            num_environments=len(scenes),
            preprocessing_fn=scene_preprocessing_fn,
            prefetch_memory_size=prefetch_memory_size,
        )

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

        scene_configs_per_scene = create_env_configs(
            env_setups=self.env_setups,
            tasks=self.tasks,
            num_configs_to_generate_per_scene=num_configs_to_generate_per_scene,
        )

        batches = []
        # Split each batched scene config into individual items (batch dim =1)
        for scene_idx, scene_cfg in enumerate(scene_configs_per_scene):
            count = int(num_configs_to_generate_per_scene[scene_idx])
            cfg_list: list[ConcurrentSceneData] = []
            for i in range(count):
                # slice global states
                orig_curr = scene_cfg.current_state
                curr = RepairsSimState(batch_dim=1)
                curr.electronics_state = [orig_curr.electronics_state[i]]
                curr.physical_state = [orig_curr.physical_state[i]]
                curr.fluid_state = [orig_curr.fluid_state[i]]
                curr.tool_state = [orig_curr.tool_state[i]]
                curr.has_electronics = orig_curr.has_electronics
                curr.has_fluid = orig_curr.has_fluid
                # sanity check: ensure single-item state
                assert curr.scene_batch_dim == 1, (
                    f"Expected batch_dim=1, got {curr.scene_batch_dim}"
                )

                orig_des = scene_cfg.desired_state
                des = RepairsSimState(batch_dim=1)
                des.electronics_state = [orig_des.electronics_state[i]]
                des.physical_state = [orig_des.physical_state[i]]
                des.fluid_state = [orig_des.fluid_state[i]]
                des.tool_state = [orig_des.tool_state[i]]
                des.has_electronics = orig_des.has_electronics
                des.has_fluid = orig_des.has_fluid
                # sanity check: ensure single-item state
                assert des.scene_batch_dim == 1, (
                    f"Expected batch_dim=1, got {des.scene_batch_dim}"
                )

                # slice voxel and diffs
                vox_init_i = scene_cfg.vox_init[i].unsqueeze(0)
                vox_des_i = scene_cfg.vox_des[i].unsqueeze(0)
                diffs_i = {
                    k: scene_cfg.initial_diffs[k][i] for k in scene_cfg.initial_diffs
                }
                diff_counts_i = scene_cfg.initial_diff_counts[i : i + 1]

                cfg_list.append(
                    ConcurrentSceneData(
                        scene=None,
                        gs_entities=None,
                        cameras=None,
                        current_state=curr,
                        desired_state=des,
                        vox_init=vox_init_i,
                        vox_des=vox_des_i,
                        initial_diffs=diffs_i,
                        initial_diff_counts=diff_counts_i,
                        scene_id=scene_idx,
                        batch_dim=1,
                        reward_history=RewardHistory(batch_dim=1),
                    )
                )
            batches.append(cfg_list)
        return batches
