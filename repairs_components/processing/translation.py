from build123d import Compound, Part, RevoluteJoint, export_gltf
import genesis as gs
import tempfile

from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
from repairs_components.geometry.base import Component
from genesis.engine.entities import RigidEntity
from repairs_components.geometry.fasteners import Fastener
from repairs_components.logic.electronics.electronics_state import ElectronicsState
from repairs_components.logic.physical_state import PhysicalState
from repairs_components.logic.fluid_state import FluidState
from repairs_components.geometry.connectors.connectors import Connector
from repairs_components.logic.electronics.component import ElectricalComponent
import numpy as np
from repairs_components.training_utils.sim_state_global import RepairsSimState
from repairs_components.logic.tools.screwdriver import Screwdriver


def translate_to_genesis_scene(
    scene: gs.Scene, b123d_assembly: Compound, sim_state: RepairsSimState
):
    assert len(b123d_assembly.children) > 0, "Translated assembly has no children"

    hex_to_name: dict[str, str] = {}
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
                export_gltf(child, gltf_path)
                mesh = gs.morphs.Mesh(file=gltf_path)

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
                mesh, surface=gs.surfaces.Plastic(*child.color.to_tuple())
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

        hex_to_name[new_entity.uid] = label
        gs_entities[label] = new_entity

    return scene, hex_to_name, gs_entities


def translate_genesis_to_python(  # translate to sim state, really.
    scene: gs.Scene,
    hex_to_name: dict[str, str],
    sim_state: RepairsSimState,
):
    """
    If a fastener is picked up by a screwdriver, get its position
    Get from the scene:
    1. All rigid body positions and rotations
    2. All contacts of electronics to electronics
    3. Whether fluid detectors register on or off.
    4. Picked up fastener tip position
    """
    # get:
    # positions of all solid bodies,
    # all emitters (leds)
    # all contacts
    # all rotations
    # all contacts specifically of electronics bodies.
    # whether fastener joints are on or off.
    # whether

    assert not isinstance(sim_state.tool_state.tool, Screwdriver) or (
        sim_state.tool_state.tool.picked_up_fastener is None
        or sim_state.tool_state.tool.picked_up_fastener_name.endswith("@fastener")
    ), "Passed name of a picked up fastener is not of that of a fastener."

    entities = scene.entities
    entity: RigidEntity  # type hint

    fastener_hole_positions = {}
    fastener_tip_pos = None

    male_connector_positions = {}
    female_connector_positions = {}

    fluid_state_pos_checks: dict[str, bool] = {}

    for entity in entities:
        real_name = hex_to_name[str(entity.uid)]

        match real_name:
            case name if name.endswith("male@connector"):
                male_connector_positions[name] = entity.get_link(
                    "connector_point"
                ).get_pos()

            case name if name.endswith("female@connector"):
                female_connector_positions[name] = entity.get_link(
                    "connector_point"
                ).get_pos()

            case name if name.endswith("@solid"):
                fastener_hole_positions[name] = get_fastener_hole_positions(entity)
                sim_state.physical_state.positions[name] = entity.get_pos()
                sim_state.physical_state.rotations[name] = entity.get_ang()

            case name if name.endswith("@liquid"):
                # TODO: check whether there is a particle in the fluid_state.positions[idx]
                # among particles = liquid.get_particles()
                # I'm quite positive there is a faster way, too.
                continue

            case name if name == sim_state.tool_state.tool.picked_up_fastener_name:
                link = next(
                    link
                    for link in entity.links
                    if link.name.endswith("_to_screwdriver")
                )
                fastener_tip_pos = link.get_pos()

            case _:
                # No match found, continue to next iteration
                continue

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


def translate_compound_to_sim_state(b123d_compound: Compound) -> RepairsSimState:
    "Get RepairsSimState from the b123d_compound, i.e. translate from build123d to RepairsSimState."
    sim_state = RepairsSimState()
    for part in b123d_compound.descendants:
        part: Part

        if part.label:  # only parts with labels are expected.
            assert "@" in part.label, "part must annotate type."
            # physical state
            if part.label.endswith("@solid"):
                sim_state.physical_state.register_body(
                    name=part.label,
                    position=part.position.to_tuple(),
                    rotation=part.rotation.to_tuple(),
                )
            elif part.label.endswith(
                "@fastener"
            ):  # collect constraints, get labels of bodies,
                # collect constraints (in build123d named joints)
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
                sim_state.physical_state.register_fastener(
                    Fastener(
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
                )
    return sim_state
