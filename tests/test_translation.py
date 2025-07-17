# deliberate test, do not remove.
from build123d import Compound
import genesis as gs
from genesis.options import SimOptions
import torch

from repairs_components.geometry.connectors.connectors import ConnectorsEnum
from repairs_components.geometry.connectors.models.europlug import Europlug
from repairs_components.processing.scene_creation_funnel import move_entities_to_pos
from repairs_components.processing.tasks import AssembleTask
from repairs_components.processing.translation import (
    get_connector_pos,
    translate_compound_to_sim_state,
    translate_genesis_to_python,
    translate_state_to_genesis_scene,
)
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.training_utils.gym_env import RepairsEnv
import pytest
from pathlib import Path


@pytest.fixture
def data_dir():
    path = Path("/workspace/test_data/")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def env_cfg():
    env_cfg = {
        "num_actions": 10,  # [x, y, z, quat_w, quat_x, quat_y, quat_z, gripper_force_left, gripper_force_right, pick_up_tool]
        "joint_names": [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ],
        "default_joint_angles": {
            "joint1": 0.0,
            "joint2": -0.3,
            "joint3": 0.0,
            "joint4": -2.0,
            "joint5": 0.0,
            "joint6": 2.0,
            "joint7": 0.79,  # no "hand" here? there definitely was hand.
            "finger_joint1": 0.04,
            "finger_joint2": 0.04,
        },
        "min_bounds": (-0.6, -0.7, -0.1),
        "max_bounds": (0.5, 0.5, 2),
    }
    return env_cfg


@pytest.fixture
def obs_cfg():
    obs_cfg = {
        "num_obs": 3,  # RGB, depth, segmentation
        "res": (64, 64),
        "use_random_textures": False,
    }
    return obs_cfg


@pytest.fixture
def reward_cfg():
    reward_cfg = {
        "success_reward": 10.0,
        "progress_reward_scale": 1.0,
        "progressive": True,  # TODO : if progressive, use progressive reward calc instead.
    }
    return reward_cfg


@pytest.fixture
def io_cfg(data_dir, env_setups_two_connectors):
    io_cfg = {
        "generate_number_of_configs_per_scene": 1,
        "dataloader_settings": {"prefetch_memory_size": 1},
        "data_dir": str(data_dir),
        "save_obs": {
            "video": False,
            "new_video_every": 1000,
            "video_len": 50,
            "voxel": False,
            "electronic_graph": False,
            "mechanics_graph": False,
            "path": str(data_dir / "obs"),
        },
        "force_recreate_data": True,  # true for testing.
        "env_setup_ids": list(range(len(env_setups_two_connectors))),
    }
    return io_cfg


@pytest.fixture(autouse=True)
def cleanup_after_test(request, scene_franka_and_two_cubes):
    yield
    test_name = request.node.name
    scene, entities = scene_franka_and_two_cubes
    scene.visualizer.cameras[0].stop_recording(
        save_to_filename=f"/workspace/RepairsComponents-v0/video_{test_name}.mp4",
        fps=60,
    )
    scene.reset()
    scene.visualizer.cameras[0].start_recording()


@pytest.fixture
def env_setups_two_connectors():
    class TwoConnectors(EnvSetup):
        def desired_state_geom(self) -> Compound:
            geom_male, _, geom_female, _ = Europlug(0).bd_geometry(
                (0, 0, 0), connected=True
            )
            return Compound(children=[geom_male, geom_female])

        def linked_groups(self) -> dict[str, tuple[list[str]]]:
            return {}

    return [TwoConnectors()]


@pytest.fixture
def command_cfg():
    command_cfg = {
        "min_bounds": [
            *(-0.8, -0.8, 0),  # XYZ position min
            *(-1.0, -1.0, -1.0, -1.0),  # Quaternion components (w,x,y,z) min
            *(0.0, 0.0),  # Gripper control min
            0.0,  # tool control min (0.5 denotes pick up tool)
        ],
        "max_bounds": [
            *(0.8, 0.8, 1.0),  # XYZ position max
            # ^note: xyz is dep
            *(1.0, 1.0, 1.0, 1.0),
            *(1.0, 1.0),  # Quaternion components (w,x,y,z) max
            1.0,  # tool control max (0.5 denotes pick up tool)
        ],
    }
    return command_cfg


@pytest.fixture
def assembly_task_geoms_two_connectors(env_setups_two_connectors):
    initial = AssembleTask().perturb_initial_state(
        env_setups_two_connectors[0].starting_state_geom(), env_size=(640, 640, 640)
    )
    desired = AssembleTask().perturb_desired_state(
        env_setups_two_connectors[0].desired_state_geom(), env_size=(640, 640, 640)
    )

    return (initial, desired)


# "will init" test skipped, since all of the downstream will use it.
@pytest.fixture
def two_connectors_env(
    env_setups_two_connectors, env_cfg, obs_cfg, io_cfg, reward_cfg, command_cfg
):
    gs.init()
    env = RepairsEnv(
        ml_batch_dim=1,
        env_setups=env_setups_two_connectors,
        tasks=[AssembleTask()],
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        io_cfg=io_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )
    return env


def test_env_setup_translation_to_state_happens(two_connectors_env: RepairsEnv):
    "Test that translation happens: electronics, mechanics bodies, fasteners."
    # electronics desired state
    desired_electronics_graph = (
        two_connectors_env.concurrent_scenes_data[0]
        .desired_state.electronics_state[0]
        .graph
    )
    desired_electronics_export_graph = (
        two_connectors_env.concurrent_scenes_data[0]
        .desired_state.electronics_state[0]
        .export_graph()
    )
    assert desired_electronics_export_graph.num_nodes == 2
    assert desired_electronics_export_graph.num_edges == 1
    assert (
        desired_electronics_graph.component_type == ConnectorsEnum.EUROPLUG.value
    ).all()
    assert (desired_electronics_graph.component_id == Europlug.model_id).all()

    # Electronics initial state
    initial_electronics_graph = (
        two_connectors_env.concurrent_scenes_data[0]
        .init_state.electronics_state[0]
        .graph
    )
    initial_electronics_export_graph = (
        two_connectors_env.concurrent_scenes_data[0]
        .init_state.electronics_state[0]
        .export_graph()
    )
    assert initial_electronics_export_graph.num_nodes == 2
    assert initial_electronics_export_graph.num_edges == 0
    assert (
        initial_electronics_graph.component_type == ConnectorsEnum.EUROPLUG.value
    ).all()
    assert (initial_electronics_graph.component_id == Europlug.model_id).all()

    # edges don't match: electronics state
    assert two_connectors_env.concurrent_scenes_data[0].initial_diff_counts == 1


def test_two_connectors_match_after_step(assembly_task_geoms_two_connectors, data_dir):
    scene = gs.Scene(SimOptions(gravity=(0, 0, 0)))  # test gravity
    _initial, desired = assembly_task_geoms_two_connectors
    desired_state = translate_compound_to_sim_state(desired, [])

    europlug = Europlug(0)
    male_europlug_path = europlug.get_path(base_dir=data_dir, male=True)
    female_europlug_path = europlug.get_path(base_dir=data_dir, male=False)
    male_name = europlug.get_name(0, male_female_both=True)
    female_name = europlug.get_name(0, male_female_both=False)
    mesh_file_names = {
        male_name: str(male_europlug_path),
        female_name: str(female_europlug_path),
    }

    scene, gs_entities = translate_state_to_genesis_scene(
        scene=scene,
        sim_state=desired_state,
        mesh_file_names=mesh_file_names,
    )
    assert male_name in gs_entities and female_name in gs_entities

    move_entities_to_pos(gs_entities=gs_entities, starting_sim_state=desired_state)

    scene.build(n_envs=1, compile_kernels=True)  # because we'll need to step

    link_idx = gs_entities[male_name].get_link("connector_point")
    assert torch.isclose(
        gs_entities[male_name].get_links_pos(link_idx),
        gs_entities[female_name].get_links_pos(link_idx),
    ).all()
    assert desired_state.electronics_state[0].export_graph()

    # test translation backwards
    (
        current_state,
        picked_up_tip_positions,
        fastener_hole_positions,
        male_connector_positions,
        female_connector_positions,
    ) = translate_genesis_to_python(scene, gs_entities, desired_state)
    assert torch.isclose(
        male_connector_positions[male_name],
        gs_entities[male_name].get_links_pos(link_idx),
    ).all()
    assert torch.isclose(
        female_connector_positions[female_name],
        gs_entities[female_name].get_links_pos(link_idx),
    ).all()
    assert torch.isclose(
        male_connector_positions[male_name],
        female_connector_positions[female_name],
    ).all()  # they should be coincident

    def test_graph_features_unchanged(current_state, desired_state):
        export_current_graph = current_state.electronics_state[0].export_graph()
        export_desired_graph = desired_state.electronics_state[0].export_graph()
        assert export_current_graph.num_nodes == export_desired_graph.num_nodes
        assert export_current_graph.num_edges == export_desired_graph.num_edges
        assert export_current_graph.num_edges == 1
        assert (
            export_current_graph.component_type == export_desired_graph.component_type
        )
        assert export_current_graph.component_id == export_desired_graph.component_id

    test_graph_features_unchanged(current_state, desired_state)

    # step and test the same.
    scene.step()

    assert torch.isclose(
        gs_entities[male_name].get_links_pos(link_idx),
        gs_entities[female_name].get_links_pos(link_idx),
    ).all()
    test_graph_features_unchanged(current_state, desired_state)


# geom utils (until moved)
def test_get_connector_pos():
    parent_pos = torch.tensor([0.5, 0.5, 0.5])
    parent_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
    rel_connector_pos = torch.tensor([0.0, 0.0, 0.3])
    assert torch.isclose(
        get_connector_pos(parent_pos, parent_quat, rel_connector_pos),
        torch.tensor([0.5, 0.5, 0.2]),
    ).all()


# TODO split this test in 3 test modules: test_translation_genesis_to_python, test_translation_compound_to_sim_state, test_translation_sim_state_to_genesis
# new tests
