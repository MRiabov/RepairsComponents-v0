from build123d import CenterOf, Compound, Part, RevoluteJoint, Unit, export_gltf, Pos
import genesis as gs
import tempfile

from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
from genesis.engine.entities import RigidEntity
import torch

from repairs_components.geometry.connectors.connectors import Connector
from repairs_components.logic.electronics.component import ElectricalComponent
from repairs_components.processing.textures import get_color_by_type, get_random_texture
from repairs_components.training_utils.sim_state_global import RepairsSimState
from repairs_components.logic.tools.screwdriver import Screwdriver
from repairs_components.geometry.fasteners import Fastener


# TODO translation from offline to genesis scene.


def translate_to_genesis_scene(
    scene: gs.Scene,
    b123d_assembly: Compound,
    sim_state: RepairsSimState,
    random_textures: bool = False,
):
    assert len(b123d_assembly.children) > 0, "Translated assembly has no children"

    gs_entities: dict[str, RigidEntity] = {}

    # translate each child into genesis entities
    for child in b123d_assembly.children:
        label = child.label
        assert label is not None and "@" in label, "Label must contain '@'"
        _, part_type = label.split("@", 1)
        part_type = part_type.lower()

        if random_textures:
            surface = get_random_texture(part_type)
        else:
            # get color by type
            surface = gs.surfaces.Plastic(color=get_color_by_type(part_type))
        if part_type in (
            "connector",
            "solid",
        ):  # ideally, have them already precompiled... But...
            tmp = tempfile.NamedTemporaryFile(suffix=label + ".gltf", delete=False)
            gltf_path = tmp.name
            tmp.close()
            if part_type == "solid":
                # move gltf to 0,0,0 on center... later will be replaced back to it's original position.
                # this is done to have a 0,0,0 origin for the mesh, which will be useful for physical state translation
                center = child.center(CenterOf.BOUNDING_BOX)
                export_gltf(
                    child.moved(Pos(-center)),
                    gltf_path,
                    unit=Unit.CM,
                )  # note: maybe glb is better.
                mesh = gs.morphs.Mesh(file=gltf_path)  # pos=center.to_tuple()
                # technically, the `pos` should not do nothing, because it will be overriden by set_pos later.
                # however it does? I don't see my object later.
            elif part_type == "connector":
                connector: Connector = sim_state.electronics_state.components[label]  # type: ignore
                mjcf = connector.get_mjcf()
                with tempfile.NamedTemporaryFile(
                    suffix=label + ".xml", mode="w", encoding="utf-8", delete=False
                ) as tmp2:
                    tmp2.write(mjcf)
                    tmp2_path = tmp2.name
                mesh = gs.morphs.MJCF(file=tmp2_path, name=label, links_to_keep=True)

            new_entity = scene.add_entity(mesh, surface=surface)
        elif part_type in ("button", "led", "switch"):
            connector: ElectricalComponent = sim_state.electronics_state[0].components[
                label
            ]  # note: [0] because can use any env - they are equal.
            mjcf = connector.get_mjcf()
            with tempfile.NamedTemporaryFile(
                suffix=label + ".xml", mode="w", encoding="utf-8", delete=False
            ) as tmp2:
                tmp2.write(mjcf)
                tmp2_path = tmp2.name
            mjcf_ent = gs.morphs.MJCF(file=tmp2_path, name=label)
            new_entity = scene.add_entity(mjcf_ent, surface=surface)
        else:
            raise NotImplementedError(
                f"Not implemented for translation part type: {part_type}"
            )

        gs_entities[label] = new_entity

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

                # are active
                constraint_a_active = joint_a.connected_to is not None
                constraint_b_active = joint_b.connected_to is not None

                # if active, get connected to names
                initial_body_a = (
                    joint_a.connected_to.parent if constraint_a_active else None
                )
                initial_body_b = (
                    joint_b.connected_to.parent if constraint_b_active else None
                )

                # collect names of bodies(?)
                assert initial_body_a.label and initial_body_a.label, (
                    "Constrained parts must be labeled"
                )

                fastener = Fastener(
                    initial_body_a=initial_body_a.label
                    if constraint_a_active
                    else None,
                    initial_body_b=initial_body_b.label
                    if constraint_b_active
                    else None,
                    constraint_a_active=constraint_a_active,
                    constraint_b_active=constraint_b_active,
                    name=part.label,
                )
                sim_state.physical_state[env_idx].register_fastener(
                    fastener.name, fastener
                )

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
