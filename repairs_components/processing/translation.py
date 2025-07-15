from typing_extensions import deprecated
from build123d import CenterOf, Compound, Part, RevoluteJoint, Unit, Pos
import genesis as gs
import tempfile

from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
from genesis.engine.entities import RigidEntity
import torch
from pathlib import Path
from repairs_components.geometry.connectors import connectors
from repairs_components.geometry.connectors.connectors import Connector
from repairs_components.training_utils.sim_state_global import RepairsSimState
from repairs_components.logic.tools.screwdriver import Screwdriver
from repairs_components.geometry.fasteners import (
    Fastener,
    get_fastener_singleton_name,
    get_singleton_fastener_save_path,
)
from torch_geometric.data import Data
import numpy as np
import torch.nn.functional as F


def translate_state_to_genesis_scene(
    scene: gs.Scene,
    # b123d_assembly: Compound,
    sim_state: RepairsSimState,
    mesh_file_names: dict[str, str],
    random_textures: bool = False,
):
    "Translate the first state to genesis scene (unbatched - this is only to populate scene.)"
    "Essentially, populate the scene with meshes."
    # assert len(b123d_assembly.children) > 0, "Translated assembly has no children"
    assert len(sim_state.physical_state[0].body_indices) > 0, (
        "Translated assembly is empty."
    )
    assert len(mesh_file_names) > 0, "No meshes provided."
    val_mesh_file_names = {  # val - validation
        k: v for k, v in mesh_file_names.items() if not k.endswith("@fastener")
    }
    assert (
        len(val_mesh_file_names) == len(sim_state.physical_state[0].body_indices)
    ), f"""Number of meshes ({len(val_mesh_file_names)}) does not match number of bodies ({len(sim_state.physical_state[0].body_indices)}).
    Mesh names: {val_mesh_file_names.keys()}
    Body names: {sim_state.physical_state[0].body_indices.keys()}
    Original (unfiltered) mesh names: {mesh_file_names.keys()}"""
    assert set(val_mesh_file_names.keys()) == set(
        sim_state.physical_state[0].body_indices.keys()
    ), f"""Mesh names do not match body indices.
    Mesh names: {val_mesh_file_names.keys()}
    Body names: {sim_state.physical_state[0].body_indices.keys()}
    Original (unfiltered) mesh names: {mesh_file_names.keys()}"""

    gs_entities: dict[str, RigidEntity] = {}

    physical_state = sim_state.physical_state[0]
    electronics_state = sim_state.physical_state[0]

    # translate each child into genesis entities
    for body_name, body_idx in physical_state.body_indices.items():
        assert body_name is not None and "@" in body_name, "Label must contain '@'"
        # note: body name is equal to build123d label here.
        part_name, part_type = body_name.split("@", 1)
        part_type = part_type.lower()

        # DEBUG: (turn this back on)
        # if random_textures:
        #     surface = get_random_texture(part_type)
        # else:
        #     # get color by type
        #     surface = gs.surfaces.Plastic(color=get_color_by_type(part_type))
        surface = gs.surfaces.Plastic(color=(1, 1, 1, 1))

        pos = physical_state.graph.position[body_idx]
        quat = physical_state.graph.quat[body_idx]
        count_fasteners_held = physical_state.graph.count_fasteners_held[body_idx]

        mesh_or_mjcf_path = str(mesh_file_names[body_name])

        if part_type == "solid" or part_type == "fixed_solid":
            fixed = part_type == "fixed_solid"
            mesh = gs.morphs.Mesh(file=mesh_or_mjcf_path, fixed=fixed)
            new_entity = scene.add_entity(mesh, surface=surface)
        elif part_type == "connector":
            # NOTE: links to keep was on UDRF, not on mjcf!!!
            mesh = gs.morphs.Mesh(file=mesh_or_mjcf_path)
            new_entity = scene.add_entity(mesh, surface=surface)
        elif part_type in ("button", "led", "switch"):
            # NOTE: links to keep was on UDRF, not on mjcf!!!
            mesh = gs.morphs.MJCF(file=mesh_or_mjcf_path)
            new_entity = scene.add_entity(mesh, surface=surface)
        elif part_type == "fastener":
            raise ValueError(
                "Fasteners should be defined only in edge attributes or free bodies, not in indices."
            )
        else:
            raise NotImplementedError(
                f"Not implemented for translation part type: {part_type}"
            )

        gs_entities[body_name] = new_entity

    singleton_fastener_morphs = {}  # cache to reduce load times.

    for fastener_id, attached_to in enumerate(
        sim_state.physical_state[0].graph.fasteners_attached_to
    ):
        fastener_d = sim_state.physical_state[0].graph.fasteners_diam[fastener_id]
        fastener_h = sim_state.physical_state[0].graph.fasteners_length[fastener_id]

        fastener_name = get_fastener_singleton_name(
            float(fastener_d), float(fastener_h)
        )
        fastener_path = mesh_file_names[fastener_name]

        # mjcf is acceptable for fasteners because it's faster(?)
        if fastener_name not in singleton_fastener_morphs:
            morph = gs.morphs.MJCF(file=str(fastener_path))
            # note^ this kind of stuff better be globally cached.
            singleton_fastener_morphs[fastener_name] = morph
        else:
            morph = singleton_fastener_morphs[fastener_name]
        new_entity = scene.add_entity(morph, surface=surface)
        # and how it will be moved later does not matter now

        # store fastener entity in gs_entities
        gs_entities[Fastener.fastener_name_in_simulation(fastener_id)] = new_entity

        # NOTE: ^ there probably is sense to store fasteners as fasteners id in gs_entities, not as length+height.
        # there is no need for length+height, but there is value in unique id.

    return scene, gs_entities


def translate_genesis_to_python(  # translate to sim state, really.
    scene: gs.Scene,
    gs_entities: dict[str, RigidEntity],
    sim_state: RepairsSimState,
    starting_hole_positions: dict[str, torch.Tensor],
    starting_hole_quats: dict[str, torch.Tensor],
    device: torch.device | None = None,
):
    """
    Get raw values from sim scene.

    Returns:
        sim_state: RepairsSimState with active bodies in the physical state.
        picked_up_tip_positions: torch.Tensor[n_envs,3] - tip pos per env
        fastener_hole_positions: dict[str, torch.Tensor] - [n_envs,3]
        male_connector_positions: dict[str, torch.Tensor] - [n_envs,3]
        female_connector_positions: dict[str, torch.Tensor] - [n_envs,3]
    """
    assert len(gs_entities) > 0, "Genesis entities can not be empty"
    # batch over all environments
    n_envs = scene.n_envs
    env_idx = torch.arange(n_envs)

    # sanity-check tool names
    assert all(
        (
            not isinstance(ts.current_tool, Screwdriver)
            or ts.current_tool.picked_up_fastener_name is None
            or ts.current_tool.picked_up_fastener_name.endswith("@fastener")
        )
        for ts in sim_state.tool_state
    ), "picked_up_fastener_name must end with @fastener"

    # prepare outputs
    fastener_hole_positions: dict[str, torch.Tensor] = {}
    male_connector_positions: dict[str, torch.Tensor] = {}
    female_connector_positions: dict[str, torch.Tensor] = {}

    # loop entities and dispatch by suffix
    for full_name, entity in gs_entities.items():
        part_name, part_type = full_name.split("@")
        if part_type in ("solid", "fixed_solid", "connector"):
            # TODO: I haven't explicitly handled fixed solids, but it may be unnecessary(?)
            # hole_pos = get_fastener_hole_positions(entity, device=device)
            # fastener_hole_positions[full_name] = hole_pos
            pos_all = entity.get_pos(env_idx)  # fixme: 0,0,0 on pos?
            quat_all = entity.get_quat(env_idx)
            for i in range(n_envs):
                sim_state.physical_state[i].update_body(
                    full_name, pos_all[i], quat_all[i]
                )
            if part_type == "connector":
                male = part_name.endswith("male")
                if male:
                    relative_connector_pos = Connector.from_name(
                        part_name
                    ).connector_pos_relative_to_center_male
                else:
                    relative_connector_pos = Connector.from_name(
                        part_name
                    ).connector_pos_relative_to_center_female
                relative_connector_pos = (
                    torch.tensor(relative_connector_pos, device=device) / 1000
                )
                scene_connector_pos = get_connector_pos(
                    pos_all, quat_all, relative_connector_pos.unsqueeze(0)
                )
                if male:
                    male_connector_positions[full_name] = scene_connector_pos
                else:
                    female_connector_positions[full_name] = scene_connector_pos
        elif part_type == "control":
            continue  # skip.
        elif part_type == "fastener":
            # get fastener pos/quat
            fastener_pos = torch.tensor(entity.get_pos(env_idx), device=device)
            fastener_quat = torch.tensor(entity.get_quat(env_idx), device=device)
            fastener_id = int(full_name.split("@")[0])  # fastener name is "1@fastener"
            for i in range(n_envs):
                sim_state.physical_state[i].graph.fasteners_loc[fastener_id] = (
                    fastener_pos[i]
                )
                sim_state.physical_state[i].graph.fasteners_quat[fastener_id] = (
                    fastener_quat[i]
                )

        # handle picked up fastener (tip)
        # fastener_tip_pos
        for env_id in range(n_envs):
            if (
                isinstance(sim_state.tool_state[env_id].current_tool, Screwdriver)
                and sim_state.tool_state[env_id].current_tool.has_picked_up_fastener
            ):
                fastener_name = sim_state.tool_state[
                    env_id
                ].current_tool.picked_up_fastener_name
                assert fastener_name is not None
                fastener_pos = gs_entities[fastener_name].get_pos(env_id)
                fastener_quat = gs_entities[fastener_name].get_quat(env_id)
                # get tip pos
                tip_pos = get_connector_pos(
                    fastener_pos,
                    fastener_quat,
                    Fastener.get_tip_pos_relative_to_center().unsqueeze(0),
                )
                sim_state.tool_state[
                    env_id
                ].current_tool.picked_up_tip_position = tip_pos

    # update holes
    sim_state = update_hole_locs(
        sim_state, starting_hole_positions, starting_hole_quats
    )  # would be ideal if starting_hole_positions, hole_quats and hole_batch were batched already.
    max_pos = sim_state.physical_state[0].graph.position.max()
    assert max_pos <= 50, f"massive out of bounds, at env_id 0, max_pos {max_pos}"
    return sim_state, male_connector_positions, female_connector_positions


def translate_compound_to_sim_state(
    batch_b123d_compounds: list[Compound], connected_bodies: list[list[str]] = []
) -> RepairsSimState:
    "Get RepairsSimState from the b123d_compound, i.e. translate from build123d to RepairsSimState."
    assert len(batch_b123d_compounds) > 0, "Batch must not be empty."
    assert all(len(compound.leaves) > 0 for compound in batch_b123d_compounds), (
        "All compounds must have children."
    )
    sim_state = RepairsSimState(batch_dim=len(batch_b123d_compounds))
    all_male_connector_def_positions = []
    all_female_connector_def_positions = []

    for env_idx in range(len(batch_b123d_compounds)):
        b123d_compound = batch_b123d_compounds[env_idx]
        env_size = np.array((640, 640, 640))
        expected_bounds_min = np.array((0, 0, 0))
        expected_bounds_max = np.array((env_size[0], env_size[1], env_size[2]))
        assert (
            np.array(tuple(b123d_compound.bounding_box().min)) + 1e-6
            >= expected_bounds_min
        ).all() and (
            np.array(tuple(b123d_compound.bounding_box().max)) - 1e-6
            <= expected_bounds_max
        ).all(), (
            f"Compound must be within bounds. Currently have {b123d_compound.bounding_box()}. Expected AABB min: {expected_bounds_min}, max: {expected_bounds_max} "
            f"\nAABBs of all parts: { {p.label: p.bounding_box() for p in b123d_compound.leaves} }"
        )
        part_dict = {part.label: part for part in b123d_compound.leaves}
        male_connector_def_positions = {}  # not tensor because we will require labels.
        female_connector_def_positions = {}  # not tensor because we will require labels.

        for part in (
            b123d_compound.leaves
        ):  # leaves, not children. children will have compounds.
            part: Part
            assert part.label, (
                f"Part must have a label. Failed at position {part.position}, volume {part.volume}"
            )
            assert "@" in part.label, "Part must annotate type."
            assert part.volume > 0, "Part must have a volume."

            part_name, part_type = part.label.split("@", 1)
            # physical state
            if part_type == "solid" or part_type == "fixed_solid":
                fixed = part_type == "fixed_solid"
                sim_state.physical_state[env_idx].register_body(
                    name=part.label,
                    # position=tuple(part.position), # this isn't always true
                    position=tuple(part.center(CenterOf.BOUNDING_BOX)),
                    rotation=tuple(part.global_location.orientation),
                    fixed=fixed,  # FIXME: orientation is as euler but I need quat!
                )
            elif part_type == "fastener":  # collect constraints, get labels of bodies,
                # collect constraints (in build123d named joints)
                assert "fastener_joint_a" in part.joints, (
                    "Fastener must have a joint a."
                )
                assert "fastener_joint_b" in part.joints, (
                    "Fastener must have a joint b."
                )
                assert "fastener_joint_tip" in part.joints, (
                    "Fastener must have a tip joint."
                )

                joint_a: RevoluteJoint = part.joints["fastener_joint_a"]
                joint_b: RevoluteJoint = part.joints["fastener_joint_b"]
                joint_tip: RevoluteJoint = part.joints["fastener_joint_tip"]

                # check if constraint_b active
                # note: constraint a may only be inactive in perturbed "disassembled" states.

                constraint_a_active = joint_a.connected_to is not None
                constraint_b_active = joint_b.connected_to is not None

                # if active, get connected to names
                initial_body_a = (
                    joint_a.connected_to.parent if constraint_a_active else None
                )
                initial_body_b = (
                    joint_b.connected_to.parent if constraint_b_active else None
                )

                fastener = Fastener(
                    initial_body_a=initial_body_a.label
                    if constraint_a_active
                    else None,
                    initial_body_b=initial_body_b.label
                    if constraint_b_active
                    else None,
                    constraint_b_active=constraint_b_active,
                )
                sim_state.physical_state[env_idx].register_fastener(fastener)

            # FIXME: no electronics implemented???
            elif part_type == "liquid":
                raise NotImplementedError("Liquid is not handled yet.")
            elif part_type == "button":
                raise NotImplementedError("Buttons are not handled yet.")
            elif part_type == "connector":
                # register the solid part of the connector
                sim_state.physical_state[env_idx].register_body(
                    name=part.label,
                    # position=tuple(part.position), # this isn't always true
                    position=tuple(part.center(CenterOf.BOUNDING_BOX)),
                    rotation=tuple(part.global_location.orientation),
                )  # BEWARE OF ROTATION AND POSITION NOT BEING TRANSLATED. (compounds don't translate pos automatically.)

                # get the connector def position
                connector_def = part_dict[part_name + "@connector_def"]
                connector_def_position = torch.tensor(
                    tuple(connector_def.center(CenterOf.BOUNDING_BOX))
                )  # ^beware!!! this must be translated as in children translation.
                if part_name.endswith("male"):
                    male_connector_def_positions[part_name] = connector_def_position
                elif part_name.endswith("female"):
                    female_connector_def_positions[part_name] = connector_def_position
            elif part_type == "connector_def":
                continue  # connector def is already registered in connectors.
            else:
                raise NotImplementedError(
                    f"Part type {part_type} not implemented. Raise from part.label: {part.label}"
                )
        all_male_connector_def_positions.append(male_connector_def_positions)
        all_female_connector_def_positions.append(female_connector_def_positions)

    # if the last is non-empty all are non-empty...
    if all_male_connector_def_positions[-1] and all_female_connector_def_positions[-1]:
        all_male_connector_def_positions = torch.stack(all_male_connector_def_positions)
        all_female_connector_def_positions = torch.stack(
            all_female_connector_def_positions
        )
        connections = connectors.check_connections(
            all_male_connector_def_positions, all_female_connector_def_positions
        )
        for env_idx, connections in enumerate(connections):
            for connection_a, connection_b in connections:
                sim_state.electronics_state[env_idx].connect(connection_a, connection_b)

    # TODO I'll do this later.
    # possibly (reasonably) we can encode XYZ and quat of connector def positions into electronics state features.
    # however this won't mean they will be returned in export_graph.

    # assert out
    max_pos = np.array(sim_state.physical_state[0].graph.position.max())
    assert np.all(np.abs(max_pos) <= (env_size / 2 / 1000)), (
        f"massive out of bounds, at env_id 0, max_pos {max_pos}"
    )

    return sim_state


def create_constraints_based_on_graph(
    env_state: RepairsSimState,
    gs_entities: dict[str, RigidEntity],
    scene: gs.Scene,
    env_idx: torch.Tensor | None = None,
):
    """Create fastener (weld) constraints based on graph. Done once in the start."""  # note: in the future could be e.g. bearing constraints. But weld constraints as fasteners for now.
    rigid_solver = scene.sim.rigid_solver
    all_base_links = {}
    if env_idx is None:
        env_idx = torch.arange(env_state.scene_batch_dim)
    # TODO: assert that number of steps is 0 - it should be only in the start. No API for this in genesis atm.

    # all_base_links= np.full(
    #     len(env_state.physical_state[0].body_indices), -1
    # )  # fill with -1
    # < note: there is some unnecessary roundtrip that makes me use the `all_base_links` dict.
    # however it doesn't matter, it's minor optimization.
    # assert all_base_links.min() >= 0, "Some base link failed to register."
    for entity_name in env_state.physical_state[0].body_indices:
        # NOTE: body_indices does not include fasteners
        entity = gs_entities[entity_name]
        all_base_links[entity_name] = entity.base_link.idx

    # gs seems to be able to hold only so many constraints, so collect all env_idx
    # for each constraint pair and pass them to genesis in a single call.
    # NOTE: untested below.

    #   key: (fastener_base_link_idx, body_base_link_idx)
    # value: list[int] env_ids where this connection exists
    connections: dict[tuple[int, int], list[int]] = {}

    for env_id in env_idx.tolist():
        state = env_state.physical_state[env_id]
        graph: Data = state.graph
        # FIXME: this kind of works but does not account for fastener constraint
        # should actually be linked to a fastener, and fastener is linked to another body.

        # genesis supports constraints on per-scene basis.

        for fastener_id, connected_to in enumerate(graph.fasteners_attached_to):
            fastener_entity = gs_entities[
                Fastener.fastener_name_in_simulation(fastener_id)
            ]
            fastener_base_link_idx = fastener_entity.base_link.idx

            # connected_to is a tensor with two elements (body_a, body_b)
            for body_id in connected_to:
                if body_id.item() == -1:
                    # Fastener not attached to anything in this position
                    continue

                body_name = env_state.physical_state[env_id].inverse_indices[
                    body_id.item()
                ]
                body_base_link_idx = all_base_links[body_name]

                key = (fastener_base_link_idx, body_base_link_idx)
                env_list = connections.setdefault(key, [])
                env_list.append(env_id)

    # Now that we have aggregated env ids for each unique fastener-body pair, add
    # the weld constraints just once per pair.
    for (fastener_idx, body_idx), env_ids in connections.items():
        # Deduplicate env_ids to be safe
        unique_env_ids = sorted(set(env_ids))  # why sorted?
        rigid_solver.add_weld_constraint(
            torch.tensor(fastener_idx).unsqueeze(0).cuda().contiguous(),
            torch.tensor(body_idx).unsqueeze(0).cuda().contiguous(),
            envs_idx=torch.tensor(unique_env_ids).cuda().contiguous(),
        )  # move to cuda because some error.
        # NOTE: unsqueeze(0) is a workaround for len of 0d tensor bug.


def reset_constraints(scene: gs.Scene):
    scene.sim.rigid_solver.joints.clear()  # maybe it'll work?


def update_hole_locs(
    current_sim_state: RepairsSimState,
    starting_hole_positions: dict[str, torch.Tensor],
    starting_hole_quats: dict[str, torch.Tensor],
):
    """Update hole locs.

    Args:
        current_sim_state (RepairsSimState): The current sim state.
        starting_hole_positions (dict[str, torch.Tensor]): The starting hole positions. [B, num_holes_per_part, 3]
        starting_hole_quats (dict[str, torch.Tensor]): The starting hole quats. [B, num_holes_per_part, 4]

    Process:
    1. Stack the starting hole positions and quats into one [B, sum(num_holes_per_part), 3] and [B, sum(num_holes_per_part), 4]
    2. Repeat the part positions and quats for each hole
    3. Calculate the hole positions and quats
    4. Set the hole positions and quats
    """
    device = tuple(starting_hole_positions.values())[0].device
    all_starting_hole_positions = torch.cat(
        list(starting_hole_positions.values()), dim=0
    ).to(device)  # [B, sum(num_holes_per_part), 3]
    all_starting_hole_quats = torch.cat(list(starting_hole_quats.values()), dim=0).to(
        device
    )
    num_holes_per_part = torch.tensor(
        [v.shape[0] for v in starting_hole_positions.values()], device=device
    ).to(device)
    hole_batch = torch.repeat_interleave(
        torch.arange(len(num_holes_per_part), device=device), num_holes_per_part
    )

    # will remove when (if) batch RepairsSimStep.
    part_pos = torch.stack(
        [phys_state.graph.position for phys_state in current_sim_state.physical_state]
    ).to(device)  # [B, P, 3]
    part_quat = torch.stack(
        [phys_state.graph.quat for phys_state in current_sim_state.physical_state]
    ).to(device, dtype=torch.float32)  # [B, P, 4]

    part_pos_batch = torch.repeat_interleave(part_pos, num_holes_per_part, dim=1)
    part_quat_batch = torch.repeat_interleave(part_quat, num_holes_per_part, dim=1)

    # note: not sure get_connector_pos will be usable with batches.
    hole_pos = get_connector_pos(
        part_pos_batch, part_quat_batch, all_starting_hole_positions.unsqueeze(0)
    )
    hole_quat = quat_multiply(part_quat_batch, all_starting_hole_quats)
    for i, phys_state in enumerate(current_sim_state.physical_state):
        phys_state.hole_positions = hole_pos[i]
        phys_state.hole_quats = hole_quat[i]
        phys_state.hole_indices_batch = hole_batch[i]
    return current_sim_state


def get_connector_pos(
    parent_pos: torch.Tensor,
    parent_quat: torch.Tensor,
    rel_connector_pos: torch.Tensor,
):
    """
    Get the position of a connector relative to its parent. Used both in translation from compound to sim state and in screwdriver offset.
    Note: expects batched inputs of rel_connector_pos, ndim=2.
    """
    return (
        parent_pos
        + rel_connector_pos
        + 2
        * torch.cross(
            parent_quat[..., 1:],
            torch.cross(parent_quat[..., 1:], rel_connector_pos, dim=-1)
            + parent_quat[..., 0:1] * rel_connector_pos,
            dim=-1,
        )
    )


def quat_multiply(
    q1: torch.Tensor, q2: torch.Tensor, normalize: bool = True
) -> torch.Tensor:
    """Quaternion multiplication, q1 ⊗ q2.
    Inputs: [..., 4] tensors where each quaternion is [w, x, y, z]
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    q = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )
    if normalize:
        q = F.normalize(q, dim=-1)
    return q


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Returns the conjugate of a quaternion [w, x, y, z]."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_angle_diff_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Returns the angle in degrees between rotations represented by q1 and q2."""
    q1 = F.normalize(q1, dim=-1)
    q2 = F.normalize(q2, dim=-1)

    q_rel = quat_multiply(quat_conjugate(q1), q2)
    w = torch.clamp(torch.abs(q_rel[..., 0]), -1.0, 1.0)  # safe acos
    angle_rad = 2.0 * torch.acos(w)
    return torch.rad2deg(angle_rad)


def are_quats_within_angle(
    q1: torch.Tensor, q2: torch.Tensor, max_angle_deg: float
) -> torch.Tensor:
    """Returns True where angular distance between q1 and q2 is ≤ max_angle_deg."""
    return quat_angle_diff_deg(q1, q2) <= max_angle_deg
