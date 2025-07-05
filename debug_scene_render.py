import genesis as gs
from genesis.engine.entities import RigidEntity

gs.init(backend=gs.gpu)

scene_1 = gs.Scene(
    show_viewer=False,
    vis_options=gs.options.VisOptions(shadow=True),
    renderer=gs.renderers.Rasterizer(),
)
scene_2 = gs.Scene(
    show_viewer=False,
    vis_options=gs.options.VisOptions(shadow=True),
    renderer=gs.renderers.Rasterizer(),
)

scene_1.add_entity(
    gs.morphs.Plane(),
)
scene_1.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)
cam = scene_1.add_camera(
    res=(640, 480),
    pos=(3.5, 0.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=30,
    GUI=False,
)
# scene 2
scene_2.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.2)))
scene_2.add_entity(
    gs.morphs.Sphere(pos=(0, 0, 1), radius=0.2),
)
scene_2.add_camera(
    pos=(1.0, 2.5, 3.5),
    lookat=(0.0, 0.0, 0.2),
    res=(256, 256),
)


scene_1.build()
scene_2.build()

for scene in [scene_1, scene_2]:
    for camera in scene.visualizer.cameras:
        camera.start_recording()

for i in range(120):
    for scene in [scene_1, scene_2]:
        scene.step()
        for camera in scene.visualizer.cameras:
            camera.render()
for scene in [scene_1, scene_2]:
    for camera in scene.visualizer.cameras:
        camera.stop_recording(
            save_to_filename="video_" + str(scene.uid) + ".mp4", fps=60
        )
# note: remove all videos with `rm video_*`
