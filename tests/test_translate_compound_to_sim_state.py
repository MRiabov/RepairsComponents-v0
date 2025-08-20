from build123d import *  # noqa: F403
import torch
from repairs_components.geometry.b123d_utils import (
    connect_fastener_to_joint,
    fastener_hole,
)
from repairs_components.geometry.connectors.models.europlug import Europlug
import pytest

from repairs_components.geometry.fasteners import Fastener
from repairs_components.logic.physical_state import compound_pos_to_sim_pos
from repairs_components.processing.translation import (
    translate_compound_to_sim_state,
    get_starting_part_holes,
)
from repairs_components.training_utils.env_setup import EnvSetup
from ocp_vscode import show
from tests.global_test_config import test_device


class TestEnv(EnvSetup):
    _positions: torch.Tensor  # note: it's not official, for test and debug only.
    _hole_loc: torch.Tensor  # for debug too.

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

        solid = solid.part.locate(
            Pos(120, 30, 30)
        )  # locate, not located to avoid copy and label issues.
        fixed_solid = fixed_solid.part.locate(Pos(150, 30, 30))
        solid_with_hole = solid_with_hole.part.locate(Pos(75, 75, 30))
        self._hole_loc = torch.tensor(
            tuple(
                (solid_with_hole.global_location * fastener_loc.locations[0]).position
            )
        )

        # Set labels before creating connections
        solid.label = "test_solid@solid"
        fixed_solid.label = "test_fixed_solid@fixed_solid"
        solid_with_hole.label = "test_solid_with_hole@solid"

        europlug_male, terminal_def, europlug_female, _ = Europlug(
            in_sim_id=0
        ).bd_geometry((60, 70, 30))
        europlug_male = europlug_male.move(Pos(60, 70, 30))
        europlug_female = europlug_female.move(Pos(60, 70, 30))
        fastener = Fastener(initial_hole_id_b=0)
        fastener_geom = fastener.bd_geometry()
        # # fastener_geom.label = "test_fastener@fastener" # labeled at bd_geometry.
        # fastener_geom = fastener_geom.locate(
        #     solid_with_hole.global_location * fastener_loc.locations[0]
        # )  # must do relocation and connection.
        connect_fastener_to_joint(fastener_geom, hole_joint)

        all_parts = [
            solid,
            fixed_solid,
            solid_with_hole,
            fastener_geom,
            europlug_male,
            europlug_female,
        ]
        self._positions = torch.tensor(
            [
                tuple(part.global_location.position)
                for part in all_parts
                if not part.label.endswith("@fastener")
            ]
        )  # filter out fastener.
        debug_compound = Compound(
            children=all_parts, joints={"fastener_joint_a": hole_joint}
        )

        return debug_compound

    @property
    def linked_groups(self) -> dict[str, tuple[list[str]]]:
        return {}  # TODO link e.g. solid with fixed solid.


@pytest.fixture(scope="session")
def test_env_geom():
    test_env = TestEnv()
    test_env.validate()
    return test_env.desired_state_geom(), test_env._positions, test_env._hole_loc


def test_translate_compound_to_sim_state(test_env_geom):
    compound, positions, hole_loc = test_env_geom
    sim_state, sim_info = translate_compound_to_sim_state([compound])
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
    assert sim_info.physical_info.fasteners_diam.shape == (expected_num_fasteners,)
    assert sim_info.physical_info.fasteners_length.shape == (expected_num_fasteners,)
    assert phys_state.fasteners_attached_to_body.shape == (
        expected_batch_dim,
        expected_num_fasteners,
        2,
    )
    assert len(sim_info.physical_info.body_indices) == expected_num_solids
    assert len(sim_info.physical_info.inverse_body_indices) == expected_num_solids

    assert phys_state.hole_positions.shape == (
        expected_batch_dim,
        expected_num_holes,
        3,
    )
    assert phys_state.hole_quats.shape == (expected_batch_dim, expected_num_holes, 4)
    assert sim_info.physical_info.part_hole_batch.shape == (expected_num_holes,)

    # assert values.
    keys = set(sim_info.physical_info.body_indices.keys())
    # Verify solids present
    assert {
        "test_solid@solid",
        "test_fixed_solid@fixed_solid",
        "test_solid_with_hole@solid",
    }.issubset(keys)
    # Verify two europlug connectors exist (IDs may vary like -1 or 0)
    europlug_connectors = [
        k for k in keys if k.endswith("@connector") and k.startswith("europlug_")
    ]
    assert len(europlug_connectors) == 2
    assert phys_state.position.allclose(
        compound_pos_to_sim_pos(positions)
    )  # note: expected incorrect X value in the 4th as it is calculated dynamically.
    assert phys_state.quat.allclose(
        torch.tensor(
            [[[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]],
            dtype=torch.float,
        )
    )
    assert phys_state.hole_quats.allclose(
        torch.tensor([[[1, 0, 0, 0]]], dtype=torch.float)
    )
    # Hole expected position = part sim position + local hole offset (identity rotation here)
    holed_body_idx = sim_info.physical_info.body_indices["test_solid_with_hole@solid"]
    local_hole_offset = torch.tensor([[0.0, 0.0, 5.0]]) / 1000.0
    expected_hole_pos_single = (
        compound_pos_to_sim_pos(positions)[holed_body_idx : holed_body_idx + 1]
        + local_hole_offset
    )
    assert phys_state.hole_positions.allclose(expected_hole_pos_single.unsqueeze(0))
    assert torch.equal(
        sim_info.physical_info.part_hole_batch, torch.tensor([holed_body_idx])
    )
    assert torch.allclose(
        sim_info.physical_info.fasteners_diam,
        torch.tensor([5.0 / 1000]),  # default Fastener diameter
    )


def test_translate_compound_to_sim_state_batch(test_env_geom):
    compound, positions, hole_loc = test_env_geom
    batch_dim = 2
    sim_state, sim_info = translate_compound_to_sim_state([compound] * batch_dim)
    expected_num_solids = 5  # solid, fixed solid, solid with hole, two connectors. Fastener is not included.
    expected_num_holes = 1
    expected_num_fasteners = 1

    phys_state = sim_state.physical_state
    # assert all shapes for batched data
    assert phys_state.batch_size == (batch_dim,)

    assert phys_state.fasteners_pos.shape == (batch_dim, expected_num_fasteners, 3)
    assert phys_state.fasteners_quat.shape == (batch_dim, expected_num_fasteners, 4)
    assert sim_info.physical_info.fasteners_diam.shape == (expected_num_fasteners,)
    assert sim_info.physical_info.fasteners_length.shape == (expected_num_fasteners,)
    assert phys_state.fasteners_attached_to_body.shape == (
        batch_dim,
        expected_num_fasteners,
        2,
    )
    assert len(sim_info.physical_info.body_indices) == expected_num_solids
    assert len(sim_info.physical_info.inverse_body_indices) == expected_num_solids

    assert phys_state.hole_positions.shape == (batch_dim, expected_num_holes, 3)
    assert phys_state.hole_quats.shape == (batch_dim, expected_num_holes, 4)
    assert sim_info.physical_info.part_hole_batch.shape == (expected_num_holes,)

    # assert values for batched data
    keys = set(sim_info.physical_info.body_indices.keys())
    assert {
        "test_solid@solid",
        "test_fixed_solid@fixed_solid",
        "test_solid_with_hole@solid",
    }.issubset(keys)
    europlug_connectors = [
        k for k in keys if k.endswith("@connector") and k.startswith("europlug_")
    ]
    assert len(europlug_connectors) == 2

    # Check batched positions and quaternions
    expected_positions = compound_pos_to_sim_pos(positions).expand(batch_dim, -1, -1)
    expected_quats = torch.tensor(
        [[[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]]
        * batch_dim,
        dtype=torch.float,
    )

    # Verify batched data: each batch should have identical values
    assert phys_state.position.allclose(expected_positions)
    assert phys_state.quat.allclose(expected_quats)
    assert phys_state.hole_quats.allclose(
        torch.tensor([[[1, 0, 0, 0]]], dtype=torch.float).expand(batch_dim, -1, -1)
    )
    # check hole translation:

    holed_body_idx = sim_info.physical_info.body_indices["test_solid_with_hole@solid"]
    local_hole_offset = torch.tensor([[0.0, 0.0, 5.0]]) / 1000.0
    expected_hole_pos = (
        compound_pos_to_sim_pos(positions)[holed_body_idx : holed_body_idx + 1]
        + local_hole_offset
    ).expand(batch_dim, -1, -1)
    assert phys_state.hole_positions.allclose(expected_hole_pos)
    assert sim_info.physical_info.part_hole_batch.equal(torch.tensor([holed_body_idx]))
    assert torch.allclose(
        sim_info.physical_info.fasteners_diam,
        torch.tensor([5.0 / 1000]),  # default diameter
    )


def test_translate_compound_records_mech_linked_groups(test_env_geom):
    compound, _positions, _hole_loc = test_env_geom
    # First translate to discover actual europlug connector names
    _sim_state0, sim_info0 = translate_compound_to_sim_state([compound])
    keys = sim_info0.physical_info.body_indices.keys()
    male_name = next(k for k in keys if k.endswith("_male@connector"))
    female_name = next(k for k in keys if k.endswith("_female@connector"))
    linked = {"mech_linked": ([male_name, female_name])}
    sim_state, sim_info = translate_compound_to_sim_state(
        [compound], linked_groups=linked
    )
    # Names recorded as provided
    assert sim_info.mech_linked_groups_names == linked["mech_linked"]
    # Indices resolved to body_indices
    idx0 = sim_info.physical_info.body_indices[male_name]
    idx1 = sim_info.physical_info.body_indices[female_name]
    assert sim_info.mech_linked_groups_indices == ([idx0, idx1],)


def test_translate_compound_invalid_linked_name_raises(test_env_geom):
    compound, _positions, _hole_loc = test_env_geom
    bad = {"mech_linked": (["does_not_exist@solid"],)}
    with pytest.raises(AssertionError):
        translate_compound_to_sim_state([compound], linked_groups=bad)


def test_get_starting_part_holes_origin_and_shift(test_device):
    # Build a solid with a through fastener hole at the top face center (like solid_with_hole)
    with BuildPart() as p1:
        b1 = Box(10, 10, 10)
        with Locations(b1.faces().filter_by(Axis.Z).sort_by(Axis.Z).last):
            _, _, _ = fastener_hole(radius=3, depth=None, id=0, build_part=p1)
    part1 = p1.part
    part1.label = "case1_solid_with_hole@solid"

    # Compound at origin
    compound1 = Compound(children=[part1])
    body_indices1 = {part1.label: 0}

    pos1, quat1, depth1, through1, batch1 = get_starting_part_holes(
        compound1, body_indices1, test_device
    )

    # Expectations: position is local [0, 0, 5] mm -> meters; quat is identity;
    # depth equals box thickness (10 mm -> 0.01 m); hole is through; batch maps to body index 0
    assert pos1.shape == (1, 3)
    assert quat1.shape == (1, 4)
    assert depth1.shape == (1,)
    assert through1.shape == (1,)
    assert batch1.shape == (1,)

    assert torch.allclose(pos1, torch.tensor([[0.0, 0.0, 5.0]]) / 1000, atol=1e-6)
    assert torch.allclose(quat1, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(depth1, torch.tensor([10.0 / 1000]), atol=1e-6)
    assert torch.equal(through1, torch.tensor([True]))
    assert torch.equal(batch1, torch.tensor([0]))

    # Build an identical solid but translate the whole part in world space
    with BuildPart() as p2:
        b2 = Box(10, 10, 10)
        with Locations(b2.faces().filter_by(Axis.Z).sort_by(Axis.Z).last):
            _, _, _ = fastener_hole(radius=3, depth=None, id=0, build_part=p2)
    part2 = p2.part.locate(Pos(10, 20, 30))
    part2.label = "case2_solid_with_hole@solid"

    compound2 = Compound(children=[part2])
    body_indices2 = {part2.label: 0}

    pos2, quat2, depth2, through2, batch2 = get_starting_part_holes(
        compound2, body_indices2, test_device
    )

    # The starting hole description is local to the part, so global translation should not change it
    assert torch.allclose(pos2, pos1, atol=1e-6)
    assert torch.allclose(quat2, quat1)
    assert torch.allclose(depth2, depth1)
    assert torch.equal(through2, through1)
    assert torch.equal(batch2, torch.tensor([0]))


if __name__ == "__main__":
    show(TestEnv().desired_state_geom())
