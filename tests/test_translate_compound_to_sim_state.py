from build123d import *
import torch
from repairs_components.geometry.b123d_utils import fastener_hole
from repairs_components.geometry.connectors.models.europlug import Europlug
import pytest

from repairs_components.geometry.fasteners import Fastener
from repairs_components.processing.translation import translate_compound_to_sim_state
from repairs_components.training_utils.env_setup import EnvSetup
from ocp_vscode import show


class TestEnv(EnvSetup):
    def desired_state_geom(self) -> Compound:
        with BuildPart() as solid:
            Box(10, 10, 10)

        with BuildPart() as fixed_solid:
            Box(10, 10, 10)

        with BuildPart() as solid_with_hole:
            box = Box(10, 10, 10)
            with Locations(box.faces().filter_by(Axis.Z).sort_by(Axis.Z).last):
                _, fastener_loc, hole_joint = fastener_hole(
                    radius=3, depth=None, id=0, build_part=solid_with_hole
                )

        solid = solid.part.located(Pos(120, 0, 30))
        fixed_solid = fixed_solid.part.located(Pos(150, 0, 30))
        solid_with_hole = solid_with_hole.part.located(Pos(75, 75, 30))

        # solid = solid.part
        # fixed_solid = fixed_solid.part
        # solid_with_hole = solid_with_hole.part
        europlug_male, connector_def, europlug_female, _ = Europlug(0).bd_geometry(
            (60, 30, 30)
        )
        fastener = Fastener(initial_hole_id_b=0)
        fastener_geom = fastener.bd_geometry()
        fastener_geom = fastener_geom.located(
            solid_with_hole.global_location * fastener_loc.locations[0]
        )  # must do relocation and connection.
        fastener_geom.joints["fastener_joint_a"].connect_to(hole_joint)
        solid.label = "test_solid@solid"
        fixed_solid.label = "test_fixed_solid@fixed_solid"
        solid_with_hole.label = "test_solid_with_hole@solid"

        all_parts = [
            solid,
            fixed_solid,
            solid_with_hole,
            fastener_geom,
            europlug_male,
            europlug_female,
        ]

        return Compound(children=all_parts, joints={"fastener_joint_a": hole_joint})

    def linked_groups(self) -> dict[str, tuple[list[str]]]:
        return {}  # TODO link e.g. solid with fixed solid.


@pytest.fixture(scope="session")
def test_env_geom():
    test_env = TestEnv()
    test_env.validate()
    return test_env.desired_state_geom()


def test_translate_compound_to_sim_state(test_env_geom):
    sim_state, starting_holes = translate_compound_to_sim_state([test_env_geom])
    expected_num_solids = 5  # solid, fixed solid, solid with hole, two connectors. Fastener is not included.
    expected_num_holes = 1
    expected_num_fasteners = 1
    expected_batch_dim = 1
    phys_state = sim_state.physical_state
    # assert all shapes
    assert phys_state.batch_size == (expected_batch_dim,)
    # FIXME: positions should be translated to sim state...

    assert phys_state.fasteners_pos.shape == (
        expected_batch_dim,
        expected_num_fasteners,
        3,
    )
    assert phys_state.fasteners_quat.shape == (
        expected_batch_dim,
        expected_num_fasteners,
        4,
    )
    assert phys_state.fasteners_diam.shape == (expected_num_fasteners,)
    assert phys_state.fasteners_length.shape == (expected_num_fasteners,)
    assert phys_state.fasteners_attached_to.shape == (
        expected_batch_dim,
        expected_num_fasteners,
    )
    assert len(phys_state.body_indices) == (expected_num_solids,)
    assert len(phys_state.inverse_body_indices) == (expected_num_solids,)

    assert phys_state.hole_positions.shape == (
        expected_batch_dim,
        expected_num_holes,
        3,
    )
    assert phys_state.hole_quats.shape == (
        expected_batch_dim,
        expected_num_holes,
        4,
    )
    assert phys_state.part_hole_batch.shape == (expected_num_holes,)
    assert phys_state.part_hole_batch.shape == (expected_num_holes,)

    # assert values.
    assert set(phys_state.body_indices.keys()) == {
        "test_solid@solid",
        "test_fixed_solid@fixed_solid",
        "test_solid_with_hole@solid_with_hole",
        "europlug_0_male@connector",
        "europlug_0_female@connector",
    }
    assert phys_state.position.allclose(
        torch.tensor([[[0, 0, 0]], [[75, 75, 75]], [[75, 30, 30]], [[60, 30, 30]]])
    )  # note: expected incorrect X value in the 4th as it is calculated dynamically.
    assert phys_state.quat.allclose(
        torch.tensor([[[1, 0, 0, 0]], [[1, 0, 0, 0]], [[1, 0, 0, 0]], [[1, 0, 0, 0]]])
    )
    assert phys_state.hole_quats.allclose(torch.tensor([[[1, 0, 0, 0]]]))
    assert phys_state.hole_positions.allclose(torch.tensor([[[0, 0, 0]]]))
    assert phys_state.part_hole_batch == torch.tensor([0])
    assert torch.allclose(
        phys_state.fasteners_diam, torch.tensor([Fastener.diameter / 1000])
    )


def test_translate_compound_to_sim_state_batch(test_env_geom):
    batch_dim = 2
    sim_state, starting_holes = translate_compound_to_sim_state(
        [test_env_geom] * batch_dim
    )
    expected_num_solids = 5  # solid, fixed solid, solid with hole, two connectors. Fastener is not included.
    expected_num_holes = 1
    expected_num_fasteners = 1

    phys_state = sim_state.physical_state
    # assert all shapes for batched data
    assert phys_state.batch_size == (batch_dim,)

    assert phys_state.fasteners_pos.shape == (
        batch_dim,
        expected_num_fasteners,
        3,
    )
    assert phys_state.fasteners_quat.shape == (
        batch_dim,
        expected_num_fasteners,
        4,
    )
    assert phys_state.fasteners_diam.shape == (batch_dim, expected_num_fasteners)
    assert phys_state.fasteners_length.shape == (batch_dim, expected_num_fasteners)
    assert phys_state.fasteners_attached_to.shape == (
        batch_dim,
        expected_num_fasteners,
    )
    assert len(phys_state.body_indices) == (expected_num_solids,)
    assert len(phys_state.inverse_body_indices) == (expected_num_solids,)

    assert phys_state.hole_positions.shape == (
        batch_dim,
        expected_num_holes,
        3,
    )
    assert phys_state.hole_quats.shape == (
        batch_dim,
        expected_num_holes,
        4,
    )
    assert phys_state.part_hole_batch.shape == (batch_dim, expected_num_holes)

    # assert values for batched data
    assert set(phys_state.body_indices.keys()) == {
        "test_solid@solid",
        "test_fixed_solid@fixed_solid",
        "test_solid_with_hole@solid_with_hole",
        "europlug_0_male@connector",
        "europlug_0_female@connector",
    }

    # Check batched positions and quaternions
    expected_positions = torch.tensor(
        [[[0, 0, 0]], [[75, 75, 75]], [[75, 30, 30]], [[60, 30, 30]]]
    )  # Single environment positions
    expected_quats = torch.tensor(
        [[[1, 0, 0, 0]], [[1, 0, 0, 0]], [[1, 0, 0, 0]], [[1, 0, 0, 0]]]
    )

    # Verify batched data: each batch should have identical values
    for b in range(batch_dim):
        assert phys_state.position[b].allclose(expected_positions)
        assert phys_state.quat[b].allclose(expected_quats)
        assert phys_state.hole_quats[b].allclose(torch.tensor([[[1, 0, 0, 0]]]))
        assert phys_state.hole_positions[b].allclose(torch.tensor([[[0, 0, 0]]]))
        assert phys_state.part_hole_batch[b] == torch.tensor([0])
        assert torch.allclose(
            phys_state.fasteners_diam[b], torch.tensor([Fastener.diameter / 1000])
        )


if __name__ == "__main__":
    show(TestEnv().desired_state_geom())
