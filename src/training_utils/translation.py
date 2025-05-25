from build123d import Compound
import genesis as gs
import tempfile
from src.geometry.base import Component
from genesis.engine.entities import RigidEntity
from src.logic.electronics.electronics_state import ElectronicsState
from src.logic.physical_state import PhysicalState
from src.logic.fluid_state import FluidState


def translate_to_genesis_scene(
    b123d_assembly: Compound, usable_parts: dict[str, Component]
):
    scene = gs.Scene()
    assert len(b123d_assembly.children) > 0, "Translated assembly has no children"

    # translate each child into genesis entities
    for child in b123d_assembly.children:
        label = child.label
        assert label is not None and "@" in label, "Label must contain '@'"
        _, part_type = label.split("@", 1)
        part_type = part_type.lower()
        if part_type == "solid":  # ideally, have them already precompiled... But...
            tmp = tempfile.NamedTemporaryFile(suffix=label + ".gltf", delete=False)
            gltf_path = tmp.name
            tmp.close()
            child.export(gltf_path)
            mesh = gs.morphs.Mesh(file=gltf_path)
            scene.add_entity(mesh, surface=gs.surfaces.Plastic(*child.color.to_tuple()))
        elif part_type in ("button", "led", "switch"):
            mjcf_model = usable_parts[label].get_mjcf()
            mjcf_ent = gs.morphs.MJCF(file=mjcf_model, name=label)
            scene.add_entity(mjcf_ent)
        else:
            raise NotImplementedError(
                f"Not implemented for translation part type: {part_type}"
            )
    return scene


def translate_genesis_to_python(
    scene: gs.Scene,
    hex_to_name: dict[str, str],
    initial_state: tuple[ElectronicsState, PhysicalState, FluidState],
):
    """Get from the scene:
    1. All rigid contacts
    2. All body rotations
    3. All contacts of electronics to electronics
    4. Whether fluid detectors register on or off.
    """
    # get:
    # positions of all solid bodies,
    # all emitters (leds)
    # all contacts
    # all rotations
    # all contacts specifically of electronics bodies.
    # whether fastener joints are on or off.
    # whether
    electronics_state, physical_state, fluid_state = initial_state

    entities: list[RigidEntity] = scene.entities
    info: dict[str, tuple] = {}

    new_electronics_state = ElectronicsState()
    new_physical_state = PhysicalState()
    new_fluid_state = FluidState()

    useful_electronics_contacts = {}
    useful_fastener_joint_status = {}
    fluid_state_pos_checks: dict[str, bool] = {}

    for entity in entities:
        contacts = entity.get_contacts()
        position = entity.get_pos()
        rotation = entity.get_ang()
        joints = entity.joints
        real_name = hex_to_name[str(entity.uid)]
        info[real_name] = (contacts, position, rotation, joints)
        if real_name in electronics_state.components:
            contacts = info[real_name][0]
            geom_a = contacts["geom_a"]
            geom_b = contacts["geom_b"]
            # Concatenate geom_a and geom_b, filter out where geom_a == name
            import numpy as np

            arr = np.concatenate([np.array(geom_a), np.array(geom_b)])
            filtered_arr = [g for g in arr if g != real_name]

            useful_electronics_contacts[real_name] = filtered_arr
        # if liquid:
        if real_name.endswith("@liquid"):
            # for /idx in fluid_state.positions:
            continue
            # TODO, check whether there is a particle in the fluid_state.positions[idx] among particles = liquid.get_particles()
            # I'm quite positive there is a faster way, too.

    return new_electronics_state, new_physical_state, new_fluid_state

    # if real_name in physical_state.joint_names:
    #     (joint_name_a, joint_name_a) = physical_state.joint_names[real_name]
    #     joint_a: RigidJoint = entity.get_joint(joint_name_a)
    # unnecessary, as this will be tracked by fastener system internally.

    # same goes for joints.
