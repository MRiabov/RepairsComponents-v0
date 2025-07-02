from build123d import CenterOf, Compound, Part, RevoluteJoint, Unit, export_gltf, Pos
import genesis as gs
import tempfile

from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
from genesis.engine.entities import RigidEntity
import torch
from pathlib import Path
from repairs_components.geometry.connectors.connectors import Connector
from repairs_components.logic.electronics.component import ElectricalComponent
from repairs_components.processing.textures import get_color_by_type, get_random_texture
from repairs_components.training_utils.sim_state_global import RepairsSimState
from repairs_components.logic.tools.screwdriver import Screwdriver
from repairs_components.geometry.fasteners import (
    Fastener,
    get_fastener_singleton_name,
    get_singleton_fastener_save_path,
)
from torch_geometric.data import Data
import numpy as np


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

        if random_textures:
            surface = get_random_texture(part_type)
        else:
            # get color by type
            surface = gs.surfaces.Plastic(color=get_color_by_type(part_type))

        pos = physical_state.graph.position[body_idx]
        quat = physical_state.graph.quat[body_idx]
        count_fasteners_held = physical_state.graph.count_fasteners_held[body_idx]

        if part_type == "solid":
            mesh_path = str(mesh_file_names[body_name])
            mesh = gs.morphs.Mesh(file=mesh_path)
            new_entity = scene.add_entity(mesh, surface=surface)

        elif part_type == "fixed_solid":  # TODO fixed solids definition in other parts.
            mesh_path = str(mesh_file_names[body_name])
            mesh = gs.morphs.Mesh(file=mesh_path, fixed=True)  # fixed!
            new_entity = scene.add_entity(mesh, surface=surface)

        elif part_type in ("connector", "button", "led", "switch"):
            mjcf_path = str(mesh_file_names[body_name])
            # FIXME: deprecate MJCF from here and use native genesis configs.
            mesh = gs.morphs.MJCF(file=mjcf_path, name=body_name, links_to_keep=True)
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
        gs_entities[f"{fastener_id}@fastener"] = new_entity

        # NOTE: ^ there probably is sense to store fasteners as fasteners id in gs_entities, not as length+height.
        # there is no need for length+height, but there is value in unique id.

    return scene, gs_entities


def translate_genesis_to_python(  # translate to sim state, really.
    scene: gs.Scene,
    gs_entities: dict[str, RigidEntity],
    sim_state: RepairsSimState,
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
            or ts.current_tool.picked_up_fastener_name.endswith("@fastener")
        )
        for ts in sim_state.tool_state
    ), "picked_up_fastener_name must end with @fastener"

    # prepare outputs
    fastener_hole_positions: dict[str, torch.Tensor] = {}
    male_connector_positions: dict[str, torch.Tensor] = {}
    female_connector_positions: dict[str, torch.Tensor] = {}
    picked_up_tip = torch.full((n_envs, 3), float("nan"))

    # loop entities and dispatch by suffix
    for full_name, entity in gs_entities.items():
        parts = full_name.split("@")
        tag2 = "@".join(parts[-2:]) if len(parts) > 1 else parts[-1]
        if tag2 == "male@connector":
            male_connector_positions[full_name] = entity.get_links_pos(env_idx)[
                "connector_point"
            ]
        elif tag2 == "female@connector":
            female_connector_positions[full_name] = entity.get_links_pos(env_idx)[
                "connector_point"
            ]
        elif tag2 == "solid":
            hole_pos = get_fastener_hole_positions(entity)
            fastener_hole_positions[full_name] = hole_pos
            pos_all = entity.get_pos(env_idx)
            ang_all = entity.get_ang(env_idx)
            for i in range(n_envs):
                sim_state.physical_state[i].register_body(
                    full_name, pos_all[i], ang_all[i]
                )
        elif tag2 == "control":
            continue
        else:
            # note: it will be more troublesome to handle a non-implemented error in batching, so whatever.
            # it should be really checked in a separate assertion.
            mask = torch.tensor(
                [
                    isinstance(ts.current_tool, Screwdriver)
                    and ts.current_tool.picked_up_fastener_name == full_name
                    for ts in sim_state.tool_state
                ],
                dtype=torch.bool,
            )
            if mask.any():
                link = next(
                    l for l in entity.links if l.name.endswith("_to_screwdriver")
                )
                pos_full = link.get_pos(env_idx)
                picked_up_tip[mask] = pos_full[mask]

    return (
        sim_state,
        picked_up_tip,
        fastener_hole_positions,
        male_connector_positions,
        female_connector_positions,
    )


def get_fastener_hole_positions(entity: RigidEntity):
    link: RigidLink
    fastener_hole_positions = {}
    for link in entity.links:
        if link.name.endswith("_hole"):  # if fastener hole... (how to get IDs?)
            fastener_hole_pos = link.get_pos()  # not sure how to get the env.
            fastener_hole_positions.update({link.name[:-3]: fastener_hole_pos})
    return fastener_hole_positions


def translate_compound_to_sim_state(
    batch_b123d_compounds: list[Compound],
) -> RepairsSimState:
    "Get RepairsSimState from the b123d_compound, i.e. translate from build123d to RepairsSimState."
    assert len(batch_b123d_compounds) > 0, "Batch must not be empty."
    assert all(len(compound.descendants) > 0 for compound in batch_b123d_compounds), (
        "All compounds must have descendants."
    )
    sim_state = RepairsSimState(batch_dim=len(batch_b123d_compounds))

    for env_idx in range(len(batch_b123d_compounds)):
        b123d_compound = batch_b123d_compounds[env_idx]
        for part in b123d_compound.descendants:
            part: Part
            assert part.label, (
                f"Part must have a label. Failed at position {part.position}"
            )
            assert "@" in part.label, "Part must annotate type."
            assert part.volume > 0, "Part must have a volume."
            # physical state
            if part.label.endswith("@solid"):
                sim_state.physical_state[env_idx].register_body(
                    name=part.label,
                    # position=part.position.to_tuple(), # this isn't always true
                    position=part.center(CenterOf.BOUNDING_BOX).to_tuple(),
                    rotation=part.location.orientation.to_tuple(),
                )
            elif part.label.endswith(
                "@fastener"
            ):  # collect constraints, get labels of bodies,
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
                constraint_b_active = joint_b.connected_to is not None

                # if active, get connected to names
                # a is always active
                initial_body_a = joint_a.connected_to.parent
                initial_body_b = (
                    joint_b.connected_to.parent if constraint_b_active else None
                )

                # collect names of bodies(?)
                assert initial_body_a.label and initial_body_a.label, (
                    "Constrained parts must be labeled"
                )

                fastener = Fastener(
                    initial_body_a=initial_body_a.label,
                    initial_body_b=initial_body_b.label
                    if constraint_b_active
                    else None,
                    constraint_b_active=constraint_b_active,
                )
                sim_state.physical_state[env_idx].register_fastener(fastener)

            # FIXME: no electronics implemented???
            elif part.label.endswith("@liquid"):
                raise NotImplementedError("Liquid is not handled yet.")
            elif part.label.endswith("@button"):
                continue
            elif part.label.endswith("@connector"):
                raise NotImplementedError("Connector is not handled yet.")
            else:
                raise NotImplementedError(f"Part type {part.label} not implemented.")

    return sim_state


def create_constraints_based_on_graph(
    env_state: RepairsSimState, gs_entities: dict[str, RigidEntity], scene: gs.Scene
):
    """Create fastener (weld) constraints based on graph. Done once in the start."""  # note: in the future could be e.g. bearing constraints. But weld constraints as fasteners for now.
    rigid_solver = scene.sim.rigid_solver
    all_base_links = {}
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

    for env_id, state in enumerate(env_state.physical_state):
        graph: Data = state.graph
        # FIXME: this kind of works but does not account for fastener constraint
        # should actually be linked to a fastener, and fastener is linked to another body.

        # genesis supports constraints on per-scene basis.

        for fastener_id, connected_to in enumerate(graph.fasteners_attached_to):
            body_a_id = connected_to[0]
            body_b_id = connected_to[1]

            fastener_entity = gs_entities[f"{fastener_id}@fastener"]
            # fastener_base_link_idx = fastener_entity.base_link.idx
            fastener_base_link_idx = fastener_entity.base_link.idx

            if body_a_id.item() != -1:
                body_a_name = env_state.physical_state[env_id].inverse_indices[
                    body_a_id.item()
                ]
                # weld the fastener to bodies in which id!=-1
                rigid_solver.add_weld_constraint(
                    fastener_base_link_idx, all_base_links[body_a_name], envs_idx=env_id
                )

            if body_b_id.item() != -1:
                body_b_name = env_state.physical_state[env_id].inverse_indices[
                    body_b_id.item()
                ]
                # weld the fastener to bodies in which id!=-1
                rigid_solver.add_weld_constraint(
                    fastener_base_link_idx, all_base_links[body_b_name], envs_idx=env_id
                )


def reset_constraints(scene: gs.Scene):
    scene.sim.rigid_solver.joints.clear()  # maybe it'll work?
