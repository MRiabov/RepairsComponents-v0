import threading
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
)
import genesis as gs
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.processing.tasks import Task


class MultiEnvDataLoader:
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

        # Track which environments are currently active/requested
        self.active_envs = set()  # TODO: remove.
        # self.env_access_history = deque(maxlen=100)  # Track recent access patterns

        # Threading components
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.prefetch_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        # Track ongoing batch preprocessing tasks
        self.pending_futures: list = []
        # Start background prefetching thread
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker, daemon=True
        )
        self.prefetch_thread.start()

        # Environment-specific state tracking
        # self.env_states = {}  # Store current state per environment

    def register_environment(self, env_idx: int, initial_state: Any = None):
        """Register an environment and optionally set its initial state."""
        if env_idx >= self.num_environments:
            raise ValueError(
                f"env_idx {env_idx} exceeds num_environments {self.num_environments}"
            )

        with self.prefetch_lock:
            self.active_envs.add(env_idx)
            if initial_state is not None:
                # self.env_states[env_idx] = initial_state
                pass
            self._trigger_prefetch()

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
            List of preprocessed data for each active environment
        """
        assert len(self.active_envs) == len(num_configs_to_generate_per_scene), (
            f"num_configs_to_generate_per_scene length ({len(num_configs_to_generate_per_scene)}) "
            f"mismatches count of registered envs ({len(self.active_envs)})."
        )
        assert num_configs_to_generate_per_scene.dtype == torch.uint16, (
            "Expected num_configs_to_generate_per_scene to be a uint16 tensor."
        )  # non-negative integer in general.

        num_configs_to_generate = torch.zeros_like(num_configs_to_generate_per_scene)
        results = []
        for scene_id, num_to_generate_tensor in enumerate(
            num_configs_to_generate_per_scene
        ):
            num_to_generate = int(num_to_generate_tensor)
            results_this_scene = []
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
            results.append(results_this_scene)

        # generator - count however much is not enough and gen it.
        new_configs = self.preprocessing_fn(num_configs_to_generate)

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
                self.prefetch_queues[scene_id].put(new_configs[scene_id])

        return results

    # def update_environment_state(self, env_idx: int, new_state: Any):
    #     """Update environment state (e.g., after reset or step)."""
    #     with self.prefetch_lock:
    #         self.env_states[env_idx] = new_state
    #         # Clear old prefetched data since state changed
    #         self._clear_prefetch_queue(env_idx)
    #         # Trigger new prefetching
    #         self._trigger_prefetch_for_env(env_idx)

    def deactivate_environment(self, env_idx: int):
        """Stop prefetching for an environment that's no longer needed."""
        with self.prefetch_lock:
            self.active_envs.discard(env_idx)
            self._clear_prefetch_queue(env_idx)
            # cancel all pending batch futures
            for future in self.pending_futures:
                future.cancel()
            self.pending_futures.clear()

    def _clear_prefetch_queue(self, env_idx: int):
        """Clear prefetch queue for an environment."""
        while not self.prefetch_queues[env_idx].empty():
            try:
                self.prefetch_queues[env_idx].get_nowait()
            except queue.Empty:
                break

    def _trigger_prefetch(self):
        """Trigger batch prefetch for all active environments."""
        if not self.active_envs:
            return

        print("trigger happens before out of mem.")

        # build tensor of per-env prefetch counts
        num = torch.zeros(self.num_environments, dtype=torch.uint16)
        for i in self.active_envs:
            free = self.prefetch_size - self.prefetch_queues[i].qsize()
            num[i] = free if free > 0 else 0
        # submit one batch job
        future = self.executor.submit(self.preprocessing_fn, num)
        self.pending_futures.append(future)

    def _prefetch_worker(self):
        """Background worker that manages prefetching based on access patterns."""
        while not self.shutdown_event.is_set():
            # process batch futures
            with self.prefetch_lock:
                done = [f for f in self.pending_futures if f.done()]
                self.pending_futures = [f for f in self.pending_futures if not f.done()]
            for f in done:
                batch = f.result()  # list of lists per scene
                for idx, items in enumerate(batch):
                    q = self.prefetch_queues[idx]
                    for it in items:
                        if not q.full():
                            q.put_nowait(it)

            with self.prefetch_lock:
                self._trigger_prefetch()

            time.sleep(0.005)  # Small delay to prevent busy waiting

    # def _intelligent_prefetch(self):  # could be any prioritized prefetch.
    #     """Intelligently prefetch based on access patterns."""
    #     if len(self.env_access_history) < 2:
    #         return

    #     # Simple pattern: prefetch for recently accessed environments
    #     recent_envs = set(list(self.env_access_history)[-5:])  # Last 5 accesses

    #     with self.prefetch_lock:
    #         for env_idx in recent_envs:
    #             if env_idx in self.active_envs:
    #                 self._trigger_prefetch_for_env(env_idx)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the dataloader."""
        with self.prefetch_lock:
            stats = {
                "active_environments": len(self.active_envs),
                "queue_sizes": {
                    env_idx: q.qsize() for env_idx, q in self.prefetch_queues.items()
                },
                "pending_tasks": len(self.pending_futures),
                # "recent_access_pattern": list(self.env_access_history)[-10:],
            }
        return stats

    def shutdown(self):
        """Clean shutdown of the dataloader."""
        self.shutdown_event.set()
        self.prefetch_thread.join(timeout=5.0)
        self.executor.shutdown(wait=True)


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

        # Initialize all scenes as active
        for i in range(len(scenes)):
            self.register_environment(i, initial_state=None)

    def _generate_scene_batches(
        self, num_configs_to_generate_per_scene: torch.Tensor
    ) -> List[ConcurrentSceneData]:
        """Generate batches for a specific scene.

        Inputs:
        num_configs_to_generate_per_scene: torch.Tensor of shape [num_tasks_per_scene] with torch_bool."""
        # FIXME: I also need a possibility of returning controlled-size batches.
        # in fact, I don't need batches in memory at all, I need individual items.
        assert len(self.env_setups) == len(self.active_envs), (
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

    # def get_batch(
    #     self, scene_idx: int, timeout: float = 5.0
    # ) -> list[ConcurrentSceneData]:
    #     """
    #     Get a single batch for a specific scene.

    #     Args:
    #         scene_idx: Scene index to get batch from
    #         timeout: Max time to wait for batch

    #     Returns:
    #         list of tuples of (scene, batch_start, batch_desired, vox_init, vox_des, initial_diff)
    #     """
    #     # Get the list of batches for this scene
    #     batches = self.get_processed_data(scene_idx, timeout=timeout)

    #     # Return a random batch from the generated batches
    #     import random

    #     return random.choice(batches)

    # def get_batch_iterator(self, scene_idx: int) -> Iterator[tuple]:
    #     """
    #     Get an iterator that yields batches for a specific scene.
    #     Automatically generates new batches when current ones are exhausted.
    #     """
    #     while True:
    #         try:
    #             batches = self.get_processed_data(scene_idx, timeout=1.0)
    #             for batch in batches:
    #                 yield batch
    #         except queue.Empty:
    #             # If no batches available, trigger immediate generation
    #             print(
    #                 f"No batches available for scene {scene_idx}, generating immediately..."
    #             )
    #             batches = self._generate_scene_batches(scene_idx)
    #             for batch in batches:
    #                 yield batch

    # note: possibly useful if there is a bug in the env, though should not be.
    # def regenerate_scene_data(self, scene_idx: int):
    #     """Trigger regeneration of data for a specific scene (e.g., after reset)."""
    #     self.update_environment_state(scene_idx, None)


# # Usage example
# if __name__ == "__main__":
#     # Mock environments for demonstration
#     class MockEnv:
#         def __init__(self, env_id):
#             self.env_id = env_id
#             self.reset()

#         def reset(self):
#             self.state = np.random.randn(5)
#             return self.state

#         def _get_obs(self):
#             return self.state

#     # Create mock environments
#     environments = [MockEnv(i) for i in range(4)]

#     # Create dataloader
#     dataloader = GenesisEnvDataLoader(
#         environments=environments,
#         preprocessing_fn=example_preprocessing_fn,
#         prefetch_size=2,
#         max_workers=2,
#     )

#     try:
#         # Simulate training loop
#         for step in range(10):
#             # Randomly select environments (simulating your RL training)
#             env_idx = np.random.choice(len(environments))

#             print(f"Step {step}: Requesting data for env {env_idx}")

#             # Get preprocessed data
#             processed_data = dataloader.get_processed_data(env_idx)
#             print(f"Got data shape: {processed_data.shape}")

#             # Simulate environment reset (happens sometimes)
#             if np.random.random() < 0.3:
#                 print(f"Environment {env_idx} reset!")
#                 environments[env_idx].reset()
#                 dataloader.on_environment_reset(env_idx)

#             # Print stats every few steps
#             if step % 3 == 0:
#                 stats = dataloader.get_stats()
#                 print(f"Stats: {stats}")

#             time.sleep(0.5)  # Simulate training time

#     finally:
#         dataloader.shutdown()
