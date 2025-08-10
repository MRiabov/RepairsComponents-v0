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
from repairs_components.processing.geom_utils import (
    euler_deg_to_quat_wxyz,
    get_connector_pos,
    quat_multiply,
    quat_conjugate,
    quat_angle_diff_deg,
    are_quats_within_angle,
    sanitize_quaternion,
)
from repairs_components.logic.physical_state import (
    register_bodies_batch,
    register_fasteners_batch,
    compound_pos_to_sim_pos,
    update_bodies_batch,
)
from repairs_components.geometry.b123d_utils import fastener_hole_info_from_joint_name


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
    assert len(sim_state.physical_state.body_indices) > 0, (
        "Translated assembly is empty."
    )
    assert len(mesh_file_names) > 0, "No meshes provided."
    val_mesh_file_names = {  # val - validation
        k: v for k, v in mesh_file_names.items() if not k.endswith("@fastener")
    }
    assert (
        len(val_mesh_file_names) == len(sim_state.physical_state.body_indices)
    ), f"""Number of meshes ({len(val_mesh_file_names)}) does not match number of bodies ({len(sim_state.physical_state.body_indices)}).
    Mesh names: {val_mesh_file_names.keys()}
    Body names: {sim_state.physical_state.body_indices.keys()}
    Original (unfiltered) mesh names: {mesh_file_names.keys()}"""
    assert set(val_mesh_file_names.keys()) == set(
        sim_state.physical_state.body_indices.keys()
    ), f"""Mesh names do not match body indices.
    Mesh names: {val_mesh_file_names.keys()}
    Body names: {sim_state.physical_state.body_indices.keys()}
    Original (unfiltered) mesh names: {mesh_file_names.keys()}"""

    gs_entities: dict[str, RigidEntity] = {}

    physical_state = sim_state.physical_state
    electronics_state = sim_state.electronics_state[0]

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
        sim_state.physical_state.fasteners_attached_to_body[0]
    ):
        fastener_d = sim_state.physical_state.fasteners_diam[0, fastener_id]
        fastener_h = sim_state.physical_state.fasteners_length[0, fastener_id]

        fastener_name = get_fastener_singleton_name(
            float(fastener_d * 1000), float(fastener_h * 1000)
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
    starting_hole_positions: torch.Tensor,  # [H, 3]
    starting_hole_quats: torch.Tensor,  # [H, 4]
    part_hole_batch: torch.Tensor,  # [H]
    device: torch.device | None = None,
):
    """
    Get raw values from sim scene.

    Returns:
        sim_state: RepairsSimState with active bodies in the physical state.
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

    # loop entities, gather body transforms for batched update; update fasteners directly
    body_names: list[str] = []
    physical_device = sim_state.physical_state.device
    # Preallocate using existing PhysicalState shapes to avoid stacking
    positions = torch.zeros_like(sim_state.physical_state.position)  # [B, N, 3]
    rotations = torch.zeros_like(sim_state.physical_state.quat)  # [B, N, 4]
    terminal_rel = torch.full(
        (positions.shape[1], 3),
        float("nan"),
        dtype=positions.dtype,
        device=physical_device,
    )  # [N, 3]
    fill_idx = 0

    for full_name, entity in gs_entities.items():
        part_name, part_type = full_name.split("@")
        if part_type in ("solid", "fixed_solid", "connector"):
            # Collect batched positions/rotations across environments
            pos_all = torch.as_tensor(
                entity.get_pos(env_idx), device=physical_device, dtype=positions.dtype
            )  # [B,3]
            quat_all = torch.as_tensor(
                entity.get_quat(env_idx), device=physical_device, dtype=rotations.dtype
            )  # [B,4]

            # Fill into preallocated tensors
            body_names.append(full_name)
            positions[:, fill_idx, :] = pos_all
            rotations[:, fill_idx, :] = quat_all

            # Terminal relative position or NaNs for non-connectors
            if part_type == "connector":
                male = part_name.endswith(
                    "_male"
                )  # dev note: "_male", not "male". is passes for female otherwise.
                connector = Connector.from_name(part_name)
                if male:
                    rel = torch.as_tensor(
                        connector.terminal_pos_relative_to_center_male / 1000,
                        dtype=positions.dtype,
                        device=physical_device,
                    )
                else:
                    rel = torch.as_tensor(
                        connector.terminal_pos_relative_to_center_female / 1000,
                        dtype=positions.dtype,
                        device=physical_device,
                    )
                terminal_rel[fill_idx] = rel
            fill_idx += 1

        elif part_type == "control":
            continue  # skip.
        elif part_type == "fastener":
            # get fastener pos/quat
            fastener_pos = torch.tensor(entity.get_pos(env_idx), device=device)
            fastener_quat = torch.tensor(entity.get_quat(env_idx), device=device)
            fastener_id = int(full_name.split("@")[0])  # fastener name is "1@fastener"
            sim_state.physical_state.fasteners_pos[:, fastener_id] = fastener_pos
            sim_state.physical_state.fasteners_quat[:, fastener_id] = fastener_quat

    # Perform a single batched update for all bodies
    if body_names:
        assert fill_idx == positions.shape[1], (
            "Number of filled bodies does not match allocated space"
        )
        positions = positions[:, :fill_idx]
        rotations = rotations[:, :fill_idx]
        terminal_rel = terminal_rel[:fill_idx]
        sim_state.physical_state = update_bodies_batch(
            sim_state.physical_state, body_names, positions, rotations, terminal_rel
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
            ].current_tool.picked_up_fastener_tip_position = tip_pos
            sim_state.tool_state[
                env_id
            ].current_tool.picked_up_fastener_quat = fastener_quat

    # update holes
    sim_state = update_hole_locs(
        sim_state, starting_hole_positions, starting_hole_quats, part_hole_batch
    )  # would be ideal if starting_hole_positions, hole_quats and hole_batch were batched already.
    max_pos = sim_state.physical_state[0].fasteners_pos.max()
    assert max_pos <= 50, f"massive out of bounds, at env_id 0, max_pos {max_pos}"
    return sim_state


def translate_compound_to_sim_state(
    batch_b123d_compounds: list[Compound], connected_bodies: list[list[str]] = []
) -> tuple[RepairsSimState, tuple]:
    "Get RepairsSimState from the b123d_compound, i.e. translate from build123d to RepairsSimState."
    assert len(batch_b123d_compounds) > 0, "Batch must not be empty."
    assert all(len(compound.leaves) > 0 for compound in batch_b123d_compounds), (
        "All compounds must have children."
    )
    n_envs = len(batch_b123d_compounds)
    sim_state = RepairsSimState(batch_dim=n_envs)

    # Compute hole data upfront from the first compound
    # (assuming all compounds in the batch have the same hole structure)
    first_compound = batch_b123d_compounds[0]
    first_body_indices = {}

    # Build body indices for the first compound to get hole data and init tensors
    body_idx = 0
    fastener_count = 0
    for part in first_compound.leaves:
        part_name, part_type = part.label.split("@", 1)
        if part_type in ("solid", "fixed_solid", "connector"):
            first_body_indices[part.label] = body_idx
            body_idx += 1
        elif part_type == "fastener":
            fastener_count += 1

    count_bodies = body_idx

    # Get hole data to populate part_hole_batch # it's recomputed but nobody cares.
    starting_holes = get_starting_part_holes(first_compound, first_body_indices)
    (
        part_holes_pos,
        part_holes_quat,
        _part_hole_depth,
        _hole_is_through,
        part_hole_batch,
    ) = starting_holes

    env_size_compound = np.array((640, 640, 640))
    expected_bounds_min = np.array((0, 0, 0))
    expected_bounds_max = np.array(
        (env_size_compound[0], env_size_compound[1], env_size_compound[2])
    )

    # Collect body and fastener data for batch processing
    all_body_names = []
    all_positions = torch.zeros((n_envs, count_bodies, 3), dtype=torch.float32)
    all_rotations = torch.zeros((n_envs, count_bodies, 4), dtype=torch.float32)
    fixed = torch.zeros((count_bodies), dtype=torch.bool)

    # gather data upfront.
    for env_idx in range(n_envs):
        b123d_compound = batch_b123d_compounds[env_idx]
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

        body_idx = 0  # Track body index for this environment
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

            # Collect body data
            if part_type in ("solid", "fixed_solid", "connector"):
                if env_idx == 0:
                    all_body_names.append(part.label)
                    fixed[body_idx] = part_type == "fixed_solid"

                # Convert position and rotation to tensors
                position_raw = torch.tensor(
                    tuple(part.global_location.position), dtype=torch.float32
                )
                position_sim = compound_pos_to_sim_pos(
                    position_raw.unsqueeze(0)
                ).squeeze(0)
                rotation_quat = euler_deg_to_quat_wxyz(
                    torch.tensor(
                        tuple(part.global_location.orientation), dtype=torch.float32
                    )
                )

                all_positions[env_idx, body_idx] = position_sim
                all_rotations[env_idx, body_idx] = rotation_quat
                body_idx += 1

            elif part_type == "liquid":
                raise NotImplementedError("Liquid is not handled yet.")
            elif part_type == "button":
                raise NotImplementedError("Buttons are not handled yet.")
            elif part_type == "terminal_def":
                continue  # terminal def is already registered in connectors.
            elif part_type == "fastener":
                continue  # would be handled in the next loop.
            else:
                raise NotImplementedError(
                    f"Part type {part_type} not implemented. Raise from part.label: {part.label}"
                )

    # Convert lists to tensors for batch processing
    assert all_body_names, "Expected to find bodies."

    # Prepare connector data for all bodies
    connector_positions_relative = torch.full(
        (count_bodies, 3), float("nan"), dtype=torch.float32
    )

    for i, name in enumerate(all_body_names):
        if name.endswith("@connector"):
            # get the terminal def position
            connector = Connector.from_name(name)
            if name.endswith("_male@connector"):
                connector_positions_relative[i] = torch.tensor(
                    connector.terminal_pos_relative_to_center_male / 1000,
                    dtype=torch.float32,
                )
            elif name.endswith("_female@connector"):
                connector_positions_relative[i] = torch.tensor(
                    connector.terminal_pos_relative_to_center_female / 1000,
                    dtype=torch.float32,
                )

    # Register all bodies at once using batch processing
    sim_state.physical_state = register_bodies_batch(
        sim_state.physical_state,
        all_body_names,
        all_positions,
        all_rotations,
        fixed,
        connector_positions_relative,
    )

    # Set part_hole_batch for all environments
    sim_state.physical_state.part_hole_batch = part_hole_batch.tile(
        (sim_state.physical_state.batch_size[0], 1)
    )
    sim_state = update_hole_locs(
        sim_state, part_holes_pos, part_holes_quat, part_hole_batch
    )

    fastener_positions = torch.zeros(
        (sim_state.physical_state.batch_size[0], fastener_count, 3), dtype=torch.float32
    )
    fastener_rotations = torch.zeros(
        (sim_state.physical_state.batch_size[0], fastener_count, 4), dtype=torch.float32
    )
    fastener_init_hole_a = torch.full(
        (sim_state.physical_state.batch_size[0], fastener_count), -1, dtype=torch.int32
    )
    fastener_init_hole_b = torch.full(
        (sim_state.physical_state.batch_size[0], fastener_count), -1, dtype=torch.int32
    )

    # FIXME 6.8.25: across env ids
    # it seems the logic above was also unfinished.
    for env_id in range(sim_state.physical_state.batch_size[0]):
        fastener_idx = 0
        fastener_names = []
        for part in b123d_compound.leaves:
            part_name, part_type = part.label.split("@", 1)
            # Collect fastener data for later processing
            if part_type == "fastener":
                fastener_positions[env_id, fastener_idx] = torch.tensor(
                    tuple(part.global_location.position), dtype=torch.float32
                )
                fastener_rotations[env_id, fastener_idx] = euler_deg_to_quat_wxyz(
                    torch.tensor(
                        tuple(part.global_location.orientation), dtype=torch.float32
                    )
                )
                # collect constraints, get labels of bodies,
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
                # joint_tip is not used but required for validation
                _ = part.joints["fastener_joint_tip"]

                # check if constraint_a and constraint_b are active
                constraint_a_active = joint_a.connected_to is not None
                constraint_b_active = joint_b.connected_to is not None

                # if active, get hole IDs from connected joint labels
                initial_hole_id_a = -1
                initial_hole_id_b = -1

                if constraint_a_active:
                    # Extract hole ID from the connected joint label
                    joint_label = joint_a.connected_to.label
                    assert joint_label.startswith("fastener_hole_")
                    initial_hole_id_a, _, _ = fastener_hole_info_from_joint_name(
                        joint_label
                    )

                if constraint_b_active:
                    # Extract hole ID from the connected joint label
                    joint_label = joint_b.connected_to.label
                    assert joint_label.startswith("fastener_hole_")
                    initial_hole_id_b, _, _ = fastener_hole_info_from_joint_name(
                        joint_label
                    )

                fastener_init_hole_a[env_id, fastener_idx] = initial_hole_id_a
                fastener_init_hole_b[env_id, fastener_idx] = initial_hole_id_b
                fastener_names.append(part_name)
                fastener_idx += 1

    # Register this body across all environments
    sim_state.physical_state = register_fasteners_batch(
        sim_state.physical_state,
        fastener_positions,
        fastener_rotations,
        fastener_init_hole_a,
        fastener_init_hole_b,
        fastener_names,
    )

    # NOTE: connectors and terminal_defs other solid bodies are not encoded in electronics, only functional components and their connections are encoded.
    # so check_connections will remain, however it will simply be an intermediate before the actual export graph.
    # alternatively (!) connectors can be encoded as edges in the non-export graph,
    # however, during export they will be removed and component nodes will be connected
    # directly. # or should I? why not encode connectors? the model will need the type of connectors, I presume.

    # # Check for connections using tensor-based connector positions
    # unnecessary yet because connectors are not encoded in electronics state.
    # if (
    #     len(sim_state.physical_state[0].male_terminal_positions) > 0
    #     and len(sim_state.physical_state[0].female_terminal_positions) > 0
    # ):
    #     for env_idx in range(n_envs):
    #         ps = sim_state.physical_state[env_idx]

    #         # Get tensor-based connector positions for this environment
    #         male_positions = ps.male_terminal_positions
    #         female_positions = ps.female_terminal_positions

    #         if male_positions.numel() > 0 and female_positions.numel() > 0:
    #             # Find connections using the updated check_connections function
    #             connection_indices = connectors.check_connections(
    #                 male_positions, female_positions
    #             )

    #             # Clear existing connections
    #             sim_state.electronics_state[env_idx].clear_connections()

    #             # Add new connections using body indices to get connector names
    #             for male_idx, female_idx in connection_indices.tolist():
    #                 # Get connector names from indices
    #                 male_body_idx = ps.male_terminal_batch[male_idx].item()
    #                 female_body_idx = ps.female_terminal_batch[female_idx].item()

    #                 # Convert body indices back to connector names
    #                 male_name = ps.inverse_body_indices[male_body_idx]
    #                 female_name = ps.inverse_body_indices[female_body_idx]

    #                 # Connect using string names for electronics_state compatibility
    #                 sim_state.electronics_state[env_idx].connect(male_name, female_name)
    #       sim_state.physical_state[env_idx] = ps

    # TODO I'll do this later.
    # possibly (reasonably) we can encode XYZ and quat of terminal def positions into electronics state features.
    # however this won't mean they will be returned in export_graph.

    # assert out # absolute because it's normalized and equal to both sides.
    assert sim_state.physical_state.position.shape[0] > 0, (
        "Positions were not registered."
    )
    max_pos = sim_state.physical_state.position.abs().max(dim=0).values
    assert (
        max_pos <= (torch.tensor(env_size_compound, device=max_pos.device) / 2 / 1000)
    ).all(), f"massive out of bounds, at env_id {max_pos.indices[0]}, max_pos {max_pos}"
    return sim_state, starting_holes


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
        state = env_state.physical_state
        # FIXME: this kind of works but does not account for fastener constraint
        # should actually be linked to a fastener, and fastener is linked to another body.

        # genesis supports constraints on per-scene basis.

        for fastener_id, connected_to in enumerate(
            state.fasteners_attached_to_body[env_id]
        ):
            fastener_entity = gs_entities[
                Fastener.fastener_name_in_simulation(fastener_id)
            ]
            fastener_base_link_idx = fastener_entity.base_link.idx

            # connected_to is a tensor with two elements (body_a, body_b)
            for body_id in connected_to:
                if body_id.item() == -1:
                    # Fastener not attached to anything in this position
                    continue

                body_name = env_state.physical_state.inverse_body_indices[
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
            torch.tensor(fastener_idx, device=state.device).unsqueeze(0).contiguous(),
            torch.tensor(body_idx, device=state.device).unsqueeze(0).contiguous(),
            envs_idx=torch.tensor(unique_env_ids, device=state.device).contiguous(),
        )
        # NOTE: unsqueeze(0) is a workaround for len of 0d tensor bug.


def reset_constraints(scene: gs.Scene):
    scene.sim.rigid_solver.joints.clear()  # maybe it'll work?


def update_hole_locs(
    current_sim_state: RepairsSimState,
    starting_hole_positions: torch.Tensor,  # [H, 3]
    starting_hole_quats: torch.Tensor,  # [H, 4]
    part_hole_batch: torch.Tensor,  # [H]
):
    """Update hole locs.

    Args:
        current_sim_state (RepairsSimState): The current sim state.
        starting_hole_positions (torch.Tensor): The starting (relative to body) hole positions. [H, 3]
        starting_hole_quats (torch.Tensor): The starting (relative to body) hole quats. [H, 4]
        part_hole_batch (torch.Tensor): The part hole batch. [H]

    Process:
    1. Stack the starting hole positions and quats into one [B, sum(num_holes_per_part), 3] and [B, sum(num_holes_per_part), 4]
    2. Repeat the part positions and quats for each hole
    3. Calculate the hole positions and quats
    4. Set the hole positions and quats
    """
    device = starting_hole_positions.device

    # will remove when (if) batch RepairsSimStep.
    part_pos = current_sim_state.physical_state.position.to(device)  # [B, P, 3]
    part_quat = current_sim_state.physical_state.quat.to(
        device, dtype=torch.float32
    )  # [B, P, 4]

    # duplicate (expand) values by batch
    part_pos_batched = part_pos[:, part_hole_batch]
    part_quat_batched = part_quat[:, part_hole_batch]

    # note: not sure get_connector_pos will be usable with batches.
    current_sim_state.physical_state.hole_positions = get_connector_pos(
        part_pos_batched, part_quat_batched, starting_hole_positions.unsqueeze(0)
    )
    current_sim_state.physical_state.hole_quats = quat_multiply(
        part_quat_batched, starting_hole_quats
    )
    return current_sim_state


# now in translation (it should be anyway)
def get_starting_part_holes(compound: Compound, body_indices: dict[str, int]):
    """Get the starting part holes as 'per part, relative to 0,0,0 position'"""
    part_holes_pos: torch.Tensor = torch.empty((0, 3))
    part_holes_quat: torch.Tensor = torch.empty((0, 4))
    part_hole_depth: torch.Tensor = torch.empty((0,))
    part_hole_is_through: torch.Tensor = torch.empty((0,), dtype=torch.bool)
    part_hole_batch: torch.Tensor = torch.empty((0,), dtype=torch.long)
    all_parts = compound.leaves
    filtered_parts = [
        part
        for part in all_parts
        if part.label.endswith(("@solid", "@fixed_solid", "@connector"))
    ]

    # not other fasteners.
    for part in filtered_parts:
        has_fastener_holes = any(
            joint.label.startswith("fastener_hole_") for joint in part.joints.values()
        )
        if has_fastener_holes:
            count_fastener_hole_joints = len(
                [
                    joint
                    for joint in part.joints.values()
                    if joint.label.startswith("fastener_hole_")
                ]
            )  # count by filtering
            fastener_hole_pos = torch.zeros(count_fastener_hole_joints, 3)
            fastener_hole_quat = torch.zeros(count_fastener_hole_joints, 4)
            fastener_hole_depths = torch.zeros(count_fastener_hole_joints)
            fastener_hole_is_through = torch.zeros(
                count_fastener_hole_joints, dtype=torch.bool
            )

            for joint in part.joints.values():
                if joint.label.startswith("fastener_hole_"):
                    id, depth, is_through = fastener_hole_info_from_joint_name(
                        joint.label
                    )
                    assert id < count_fastener_hole_joints, (
                        "id of joint is out of bounds. "
                        f"id: {id}, count: {count_fastener_hole_joints}"
                    )
                    fastener_hole_pos[id] = (
                        torch.tensor(tuple(joint.relative_location.position)) / 1000
                    )
                    fastener_hole_quat[id] = euler_deg_to_quat_wxyz(
                        torch.tensor(tuple(joint.relative_location.orientation))
                    )
                    fastener_hole_depths[id] = depth / 1000
                    # FIXME: I explicitly don't need depth when I have it, I need it when the hole is through.
                    fastener_hole_is_through[id] = is_through
            # note: there could be positional mismatch in id, however I hope this won't be an issue.

            # simply cat it because it's easier.
            part_holes_pos = torch.cat([part_holes_pos, fastener_hole_pos], dim=0)
            part_holes_quat = torch.cat([part_holes_quat, fastener_hole_quat], dim=0)
            part_hole_depth = torch.cat([part_hole_depth, fastener_hole_depths], dim=0)
            part_hole_is_through = torch.cat(
                [part_hole_is_through, fastener_hole_is_through], dim=0
            )
            part_hole_batch = torch.cat(
                [
                    part_hole_batch,
                    torch.full(
                        (count_fastener_hole_joints,),
                        body_indices[part.label],
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )
    # TODO sort part_hole_batch, although I don't know if that's necessary.
    return (
        part_holes_pos,
        part_holes_quat,
        part_hole_depth,
        part_hole_is_through,
        part_hole_batch,
    )
