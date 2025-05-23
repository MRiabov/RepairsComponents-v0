"""Example demonstrating a simple button simulation using the Genesis simulator."""

import genesis as gs  # type: ignore
import numpy as np


def main() -> None:
    """Run a simple button simulation and record to video."""
    # Initialize Genesis with CPU backend
    gs.init(backend=gs.cpu, logging_level="info")  # type: ignore

    # Create a headless scene (no viewer)
    scene = gs.Scene(
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
        # gravity=(0, 0, -9.81)  # Add gravity for more realistic simulation
        # gravity already exists
    )

    # Add a plane as the ground
    _plane = scene.add_entity(gs.morphs.Plane())

    # Add a button (a small cylinder that can be pressed)
    button = scene.add_entity(
        gs.morphs.Cylinder(radius=0.2, height=0.1, pos=(0, 0, 0.05)),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Plastic(
            color=(1, 0, 0, 1),  # red
        ),
    )

    # Add a box that will press the button
    box = scene.add_entity(
        gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(0, 1.0, 0.2)),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Plastic(
            color=(0, 0.5, 1, 1),  # blue
        ),
    )

    # Add a simple box
    _box = scene.add_entity(
        gs.morphs.Box(size=[0.2, 0.2, 0.2], pos=[0, 0, 0.5], quat=[1, 0, 0, 0])
    )

    # Add a camera to record the scene with a better view
    camera = scene.add_camera(
        res=(640, 480),
        pos=(1.5, -1.5, 1.5),  # Diagonal view of the scene
        lookat=(0, 0, 0.1),  # Focus on the button
        fov=45,
        GUI=False,
    )

    # Build the scene
    scene.build()

    # Start recording
    output_file_name = "simulation_video.mp4"
    output_file = "/workspace/RepairsComponents-v0/video.mp4"
    camera.start_recording()
    print(f"Starting simulation and recording to {output_file_name}...")

    # Run the simulation for a fixed number of steps
    for i in range(300):  # 300 steps at 60 FPS = 5 seconds of video
        # Apply a small force to the box to make it move towards the button
        # if i < 100:  # Only apply force for the first 100 steps
        #     box.add_force((0, -0.5, 0))  # Push the box towards the button

        # Update the camera to follow the action
        if i > 50:  # Start moving camera after a short delay
            camera.set_pose(
                # pos=(box_pos[0] + 1.5, box_pos[1] - 1.5, 1.5),  # Keep relative position to box
                # lookat=box.get_pos().cpu()  # Keep looking at the box
            )
        # note: I don't know how to make the camera follow the box; the box.get_pos returns a numpy array which seems to be static - it does not update throughout the rollout.

        scene.step()
        # Render the current frame
        camera.render()

    # Save the recording
    camera.stop_recording(save_to_filename=output_file, fps=60)
    print(f"Simulation complete. Video saved to {output_file_name}")


if __name__ == "__main__":
    main()
