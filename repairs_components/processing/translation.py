from build123d import CenterOf, Compound, Part, RevoluteJoint, Unit, export_gltf, Pos
import genesis as gs
import tempfile

from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
from genesis.engine.entities import RigidEntity
from repairs_components.geometry.connectors.connectors import Connector
from repairs_components.logic.electronics.component import ElectricalComponent
import numpy as np
from repairs_components.processing.voxel_export import PART_TYPE_COLORS
from repairs_components.training_utils.sim_state_global import RepairsSimState
from repairs_components.logic.tools.screwdriver import Screwdriver
from repairs_components.geometry.fasteners import Fastener


def translate_to_genesis_scene(
    scene: gs.Scene, b123d_assembly: Compound, sim_state: RepairsSimState
):
    assert len(b123d_assembly.children) > 0, "Translated assembly has no children"

    gs_entities: dict[str, RigidEntity] = {}

    # translate each child into genesis entities
    for child in b123d_assembly.children:
        label = child.label
        assert label is not None and "@" in label, "Label must contain '@'"
        _, part_type = label.split("@", 1)
        part_type = part_type.lower()
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
                )
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
            new_entity = scene.add_entity(
                mesh,
                surface=gs.surfaces.Plastic(
                    color=tuple(PART_TYPE_COLORS[part_type][:3])
                ),
            )
        elif part_type in ("button", "led", "switch"):
            connector: ElectricalComponent = sim_state.electronics_state.components[
                label
            ]  # type: ignore
            mjcf = connector.get_mjcf()
            with tempfile.NamedTemporaryFile(
                suffix=label + ".xml", mode="w", encoding="utf-8", delete=False
            ) as tmp2:
                tmp2.write(mjcf)
                tmp2_path = tmp2.name
            mjcf_ent = gs.morphs.MJCF(file=tmp2_path, name=label)
            new_entity = scene.add_entity(mjcf_ent)
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
    If a fastener is picked up by a screwdriver, get its position
    Get from the scene:
    1. All rigid body positions and rotations
    2. All contacts of electronics to electronics
    3. Whether fluid detectors register on or off.
    4. Picked up fastener tip position

    TODO: this docstring is pointless.
    """
    assert len(gs_entities) > 0, "Genesis entities can not be empty"
    # get:
    # positions of all solid bodies,
    # all emitters (leds)
    # all contacts
    # all rotations
    # all contacts specifically of electronics bodies.
    # whether fastener joints are on or off.
    # whether
    for env_idx in range(scene.n_envs):
        assert not isinstance(
            sim_state.tool_state[env_idx].current_tool, Screwdriver
        ) or (
            sim_state.tool_state[env_idx].current_tool.picked_up_fastener is None
            or sim_state.tool_state[
                env_idx
            ].current_tool.picked_up_fastener_name.endswith("@fastener")
        ), "Passed name of a picked up fastener is not of that of a fastener."

        entities = scene.entities
        entity: RigidEntity  # type hint

        fastener_hole_positions = {}
        fastener_tip_pos = None

        male_connector_positions = {}
        female_connector_positions = {}

        fluid_state_pos_checks: dict[str, bool] = {}

        # NOTE for future: genesis is in meters, while build123d and physical state is in cm.
        # xyz seems to be equal.

        for entity_name, entity in gs_entities.items():
            match entity_name:
                case name if name.endswith("male@connector"):
                    # NOTE: I had some issues with link.pos because it was not updated(!). So be careful and use get_link_pos in case of issues instead.
                    male_connector_positions[name] = entity.get_links_pos(env_idx)[
                        "connector_point"
                    ]  # note: this does not work and I know, but maybe there's a better way to get index.
                case name if name.endswith("female@connector"):
                    # NOTE: I had some issues with link.pos because it was not updated(!). So be careful and use get_link_pos in case of issues instead.
                    female_connector_positions[name] = entity.get_links_pos(env_idx)[
                        "connector_point"
                    ]

                case name if name.endswith("@solid"):
                    fastener_hole_positions[name] = get_fastener_hole_positions(entity)
                    sim_state.physical_state[env_idx].register_body(
                        name, entity.get_pos(env_idx), entity.get_ang(env_idx)
                    )

                case name if name.endswith("@liquid"):
                    # TODO: check whether there is a particle in the fluid_state.positions[idx]
                    # among particles = liquid.get_particles()
                    # I'm quite positive there is a faster way, too.
                    raise NotImplementedError

                case name if (
                    isinstance(sim_state.tool_state[env_idx].current_tool, Screwdriver)
                    and name
                    == sim_state.tool_state[
                        env_idx
                    ].current_tool.picked_up_fastener_name
                ):
                    link = next(
                        link
                        for link in entity.links
                        if link.name.endswith("_to_screwdriver")
                    )
                    fastener_tip_pos = link.get_pos()

                case name if name.endswith("@control"):  # ignore some here
                    continue
                case _:
                    raise NotImplementedError(
                        f"Entity {entity_name} not implemented for translation."
                    )

        return (
            sim_state,
            fastener_tip_pos,
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
