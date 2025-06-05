import logging
import genesis as gs
from genesis.vis.camera import Camera

from repairs_components.processing.scene_creation_funnel import create_env_configs
from repairs_components.processing.tasks import AssembleTask
from repairs_components.geometry.base_env.tooling_stand_plate import render_and_save
from examples.box_to_pos_task import MoveBoxSetup


def main():
    # Create an empty Genesis scene
    scene = gs.Scene(
        show_viewer=False, vis_options=gs.options.VisOptions(world_frame_size=2)
    )

    # Create an instance of AssembleTask
    task = AssembleTask()

    # Create an instance of the environment setup
    env_setup = MoveBoxSetup()

    # Create random scenes
    (
        first_desired_scene,
        cameras,
        gs_entities,
        voxel_grids_initial,
        voxel_grids_desired,
        starting_sim_states,
        desired_sim_states,
    ) = create_env_configs(scene, env_setup, [task], num_scenes_per_task=1)

    # Set up cameras for visualization
    # Render and save the scene
    render_and_save(first_desired_scene, cameras[0], cameras[1])
    print("Rendering completed. Check the 'renders' directory for output images.")


if __name__ == "__main__":
    gs.init(backend=gs.cuda)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    b123d_logger = logging.getLogger("build123d")
    b123d_logger.disabled = True

    main()
