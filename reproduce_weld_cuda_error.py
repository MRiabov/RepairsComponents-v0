import torch
import genesis as gs

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

box_1 = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.3), pos=(0.0, 0.0, 0.5)))
box_2 = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.3), pos=(0.0, 0.5, 0.5)))
cam = scene.add_camera(
    pos=(1.0, 2.5, 3.5),
    lookat=(0.0, 0.0, 0.2),
    res=(256, 256),
)

# Finalise scene – this allocates GPU buffers etc.
scene.build(n_envs=4)
device = torch.device("cuda:0")

for i in range(3):
    scene.step()
    cam.render()


env_idx_1 = torch.tensor([0], device=device, dtype=torch.int64)

scene.sim.rigid_solver.add_weld_constraint(
    torch.tensor(9999).unsqueeze(0),
    torch.tensor(9999).unsqueeze(0),
    env_idx_1,
)

scene.sim.rigid_solver.add_weld_constraint(
    torch.tensor(9999).unsqueeze(0),
    torch.tensor(9999).unsqueeze(0),
    env_idx_1,
)
