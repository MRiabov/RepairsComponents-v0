import torch
import genesis as gs
from genesis.engine.entities import RigidEntity

"""
More advanced reproducible case requested by user:

1. Replace the simple box _arm_ with an MJCF Franka Panda arm.
2. Use inverse kinematics so that the Franka "hand" link is positioned at the
   location of a second box (box_2).
3. Weld-constrain the Franka hand link to box_2.
4. Weld-constrain an independent sphere to box_2 as well.

If the illegal CUDA access bug is related to multiple welds and/or robot links
this script should surface it.
"""

# ---------------------------------------------------------------------------
# 1.  Genesis initialisation
# ---------------------------------------------------------------------------
gs.init(backend=gs.gpu)

# ---------------------------------------------------------------------------
# 2.  Scene creation
# ---------------------------------------------------------------------------
scene = gs.Scene(
    show_viewer=False,
    vis_options=gs.options.VisOptions(env_separate_rigid=True, shadow=True),
)

# Static ground
scene.add_entity(gs.morphs.Plane())

# A target box that we will weld to
box_target = scene.add_entity(
    gs.morphs.Box(size=(0.05, 0.05, 0.3), pos=(0.4, 0.0, 0.5))
)

# A free sphere that will later be welded to the target box
sphere_tool = scene.add_entity(gs.morphs.Sphere(radius=0.05, pos=(0.6, 0.0, 0.5)))

# Franka Emika Panda robot arm (MJCF)
franka: RigidEntity = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0.0, 0.0, 0.0))
)

# Minimal camera for debugging (headless render ok)
cam = scene.add_camera(pos=(2.0, 2.0, 2.0), lookat=(0, 0, 0.5), res=(320, 240))

# Build with a single environment for clarity
scene.build(n_envs=4)

device = gs.device  # typically torch.device('cuda:0')

# env_idx = torch.tensor([0], device=device, dtype=torch.int32)
env_idx = torch.nonzero(
    torch.norm(
        box_target.get_pos() - sphere_tool.get_pos(),
        dim=1,
    )
    < 5 & torch.tensor([True, True, True, True], dtype=torch.bool, device=device)
).squeeze(1)


# ---------------------------------------------------------------------------
# 3.  Move Franka hand to the box via IK
# ---------------------------------------------------------------------------
# Target pose – same as box_target
hand_target_pos = torch.tensor(box_target.get_pos(envs_idx=env_idx), device=device)
hand_target_quat = torch.tensor(box_target.get_quat(envs_idx=env_idx), device=device)

# Solve IK (returns joint positions)
q_desired = franka.inverse_kinematics(
    link=franka.get_link("hand"), pos=hand_target_pos, quat=hand_target_quat
)

# Drive joints to IK solution and step a few frames to settle
for _ in range(3):
    franka.control_dofs_position(q_desired)
    scene.step()

# ---------------------------------------------------------------------------
# 4.  Attach the *sphere_tool* to the Franka hand using the same helper that the
#     main codebase employs (`attach_tool_to_arm`).  This will internally add a
#     weld constraint exactly like in training.
# ---------------------------------------------------------------------------
from repairs_components.logic.tools.tool import attach_tool_to_arm, detach_tool_from_arm

attach_tool_to_arm(scene, sphere_tool, franka, env_idx)
scene.step()
scene.reset()
# call second time - maybe duplicate index is wrong.
attach_tool_to_arm(scene, sphere_tool, franka, env_idx)
scene.step()
scene.reset()
# call third time - maybe duplicate index is wrong.
attach_tool_to_arm(scene, sphere_tool, franka, torch.arange(3))
scene.step()
scene.reset()
#detach and attach back
detach_tool_from_arm(scene, sphere_tool, franka, env_idx)
scene.step()
scene.reset()
attach_tool_to_arm(scene, sphere_tool, franka, torch.arange(3))


print(
    "`attach_tool_to_arm` called - stepping a few frames to trigger any potential crash ..."
)
for i in range(3):
    scene.step()
    cam.render()

print(
    "Script finished without crashing - if no CUDA error appeared, the bug did not reproduce."
)
