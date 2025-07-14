import torch
import genesis as gs
from repairs_components.geometry.fasteners import Fastener
from repairs_components.processing.scene_creation_funnel import move_entities_to_pos
from repairs_components.training_utils.sim_state_global import RepairsSimState
from genesis.engine.entities import RigidEntity
from repairs_sim_step import create_constraints_based_on_graph

"""
Minimum reproducible example for CUDA illegal memory access that occurs when
calling `scene.sim.rigid_solver.add_weld_constraint(...)` with link indices that
have been `unsqueeze(0)`-ed.

Steps reproduced here:
1. Create a single `gs.Scene` with a plane ground and two primitive rigid
   entities (box and sphere).
2. Build the scene.
3. Move the *tool* entity to the exact pose of the *arm* entity, similar to the
   logic in `attach_tool_to_arm`.
4. Call `add_weld_constraint` with `unsqueeze(0)` on link indices. This should
   trigger the same CUDA error that was observed in the full training script.

Run with e.g.:
    python reproduce_weld_cuda_error.py
"""

# ----------------------------------------------------------------------------
# 1. Initialise Genesis on the GPU
# ----------------------------------------------------------------------------
gs.init(
    backend=gs.gpu, logging_level="debug"
)  # ensures Taichi / CUDA backend is selected

# ----------------------------------------------------------------------------
# 2. Build a very small scene with just ground + 2 objects
# ----------------------------------------------------------------------------
scene = gs.Scene(
    show_viewer=False,
    vis_options=gs.options.VisOptions(env_separate_rigid=True, shadow=True),
)

# Ground plane so objects have a reference
scene.add_entity(gs.morphs.Plane())

box_1: RigidEntity = scene.add_entity(
    gs.morphs.Box(size=(0.05, 0.05, 0.3), pos=(0.0, 0.0, 0.5))
)
box_2: RigidEntity = scene.add_entity(
    gs.morphs.Box(size=(0.05, 0.05, 0.3), pos=(0.0, 0.5, 0.5))
)
fastener = scene.add_entity(
    gs.morphs.MJCF(
        file="/workspace/data/shared/fasteners/fastener_d5.00_l15.00@fastener.xml"
    )
)
cam = scene.add_camera(
    pos=(1.0, 2.5, 3.5),
    lookat=(0.0, 0.0, 0.2),
    res=(256, 256),
)
sim_state = RepairsSimState(1)
phys_state = sim_state.physical_state[0]
phys_state.register_body("box", (0, 0, 0), (0, 0, 0))
phys_state.register_body(
    "box_2", (1000, 0, 0), (0, 0, 0)
)  # note: 1000 because 1/1000 later.
phys_state.register_fastener(
    Fastener(initial_body_a="box_2", initial_body_b="box", constraint_b_active=True)
)
phys_state.connect_fastener_to_one_body(0, "box_2")
phys_state.connect_fastener_to_one_body(0, "box")


# Finalise scene – this allocates GPU buffers etc.
scene.build(n_envs=1)
device = torch.device("cuda:0")

env_idx_1 = torch.tensor([0], device=device, dtype=torch.int64)

gs_entities = {"box": box_1, "box_2": box_2, "0@fastener": fastener}
move_entities_to_pos(gs_entities, sim_state)
create_constraints_based_on_graph(sim_state, gs_entities, scene, env_idx_1)


for i in range(2):
    scene.step()
    cam.render()
    scene.reset()

gs_entities = {"box": box_1, "box_2": box_2, "0@fastener": fastener}
move_entities_to_pos(gs_entities, sim_state)
create_constraints_based_on_graph(sim_state, gs_entities, scene, env_idx_1)
