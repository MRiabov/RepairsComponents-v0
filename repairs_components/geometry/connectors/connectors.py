from enum import IntEnum
from pathlib import Path
import sys
import os
from typing_extensions import deprecated

from build123d import (
    Axis,
    CenterOf,
    Compound,
    Part,
    Rotation,
    Vector,
    VectorLike,
    Color,
)
from build123d.geometry import Location
from genesis import gs
from websockets.typing import Origin
from repairs_components.logic.electronics.component import (
    ElectricalComponent,
    ElectricalComponentsEnum,
)
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Literal, Mapping
import trimesh


class Connector(ElectricalComponent):
    in_sim_id: int
    "The connector id in the simulation."
    # NOTE: this should correspond to the physical sim state connector_pos male/female tensors. It currently does not. (P2)

    _connector_def_size: float = 0.3  # vis only

    def __init__(self, in_sim_id: int):
        super().__init__(self.get_name(in_sim_id, None))
        self.in_sim_id = in_sim_id  # useful for female and male connector namesF

    def propagate(self, voltage: float, current: float) -> tuple[float, float]:
        return voltage, current  # pass through.

    @property
    def component_type(self) -> int:
        return ElectricalComponentsEnum.CONNECTOR.value

    @property
    @abstractmethod
    def model_id(self) -> int:
        """Get the model id of the connector."""
        raise NotImplementedError

    def get_name(self, in_sim_id: int, male_female_both: bool | None) -> str:
        """Get name. If male_female_both is None, return compound, name, if True, return name_male, if False, return name_female"""
        if male_female_both is None:
            male_or_female = ""
        else:
            male_or_female = "_male" if male_female_both else "_female"  # need the "_"
        return (
            f"{ConnectorsEnum(self.model_id).name.lower()}_{in_sim_id}{male_or_female}"
        )

    @property
    @abstractmethod
    def connector_pos_relative_to_center_male(self) -> np.ndarray:
        """Get the position of the connector def relative to the center of the component."""
        raise NotImplementedError

    @property
    @abstractmethod
    def connector_pos_relative_to_center_female(self) -> np.ndarray:
        """Get the position of the connector def relative to the center of the component."""
        raise NotImplementedError

    @property
    def connected_at_angle(self) -> tuple[float, float, float]:
        """Get the angle at which the connector is connected."""
        return (0, 0, 180)

    @deprecated(
        "Using MJCF in simulation is deprecated as unnecessary. Use meshes directly instead."
    )
    def get_mjcf(
        self,
        base_dir: Path,
        male: bool,
        connector_position: np.ndarray | None = None,
        density: float = 1000,
    ):
        """This defines a generic mjcf for a connector with a connector position."""
        if connector_position is None:
            connector_position = (
                self.connector_pos_relative_to_center_male
                if male
                else self.connector_pos_relative_to_center_female
            )
        print(
            "warning: exporting electronics as mjcf is deprecated; replace with native genesis controls."
        )
        name = self.get_name(self.in_sim_id, male)
        mesh_file_path = str(
            self.save_path_from_name(base_dir, name, "obj")
        )  # not the bad code to care of.
        assert Path(mesh_file_path).exists(), (
            f"Mesh file {mesh_file_path} does not exist. Expected obj file."
        )
        # load mesh, compute mass and inertia.
        mesh = trimesh.load(mesh_file_path)
        mesh.density = density
        mass_properties = mesh.mass_properties
        mesh_inertia = mass_properties.inertia

        return f"""<mujoco>
        <asset>
            <mesh name="{name}" file="{mesh_file_path}"/>
        </asset>
        
        <worldbody>
            <body name="{name}">
                <geom name="{name}_geom" type="mesh" mesh="{name}" density="{mesh_mass}"/>
                
                <body name="connector_point" pos="{connector_position[0]} {connector_position[1]} {connector_position[2]}">
                    <!-- No inertial, geom, or visual tags = fixed frame -->
                </body>
            </body>
        </worldbody>
    </mujoco>
    """

    def bd_geometry(
        self,
        moved_to_male: VectorLike,
        moved_to_female: VectorLike | None = None,
        connected=True,
    ) -> tuple[Part | Compound, np.ndarray, Part | Compound, np.ndarray | None]:
        """
        return build123d geometry of containers, colored as the connector color and with connector metadata.
        """
        if not connected:
            assert moved_to_female is not None, (
                "When labelled as disconnected, expect that the moved_to_female is populated."
            )
        else:
            assert moved_to_female is None, (
                "When labelled as connected, expect that the moved_to_female is None."
            )
            moved_to_female = (0, 0, 0)  # init empty moved_to_female.

        geom_male = self.bd_geometry_male(moved_to_male)
        geom_female = self.bd_geometry_female(moved_to_female)

        # TODO: get collision detection position from the geometry.
        # get the aabb of a CHILD sphere. also discard that sphere from the geometry.

        # get the aabb of a child sphere.
        male_connector_collision_detection_position = geom_male.children[-1].center(
            CenterOf.BOUNDING_BOX
        )
        female_connector_collision_detection_position = geom_female.children[-1].center(
            CenterOf.BOUNDING_BOX
        )

        if connected:
            # TODO: move geom_male into the position where connector positions would be equal,
            # with connected_at_angle difference between them.
            # Step 1: Rotate around female_connector_collision_detection_position
            # get the index of the non-zero element -
            # rotate.
            geom_male = Rotation(self.connected_at_angle) * geom_male

            # NOTE: translate the CHILDREN, not just the parent.
            # note: but why? was this before I knew of global_location? anyway, this works.
            geom_male = Compound(
                children=[child.rotate(Axis.Z, 180) for child in geom_male.children],
                label=geom_male.label,
            )

            male_connector_collision_detection_position = geom_male.children[-1].center(
                CenterOf.BOUNDING_BOX
            )

            # Step 2: Translate to match female connector position
            pos_diff = (
                female_connector_collision_detection_position
                - male_connector_collision_detection_position
            )

            geom_male = geom_male.moved(Location(pos_diff))

        return (
            geom_male,
            male_connector_collision_detection_position,
            geom_female,
            female_connector_collision_detection_position if not connected else None,
        )

    @abstractmethod
    def bd_geometry_male(self, moved_to: VectorLike) -> Part | Compound:
        "Returns: A geometrical part, a connector markup part, and a numpy array of connector collision detection positions"
        pass

    @abstractmethod
    def bd_geometry_female(self, moved_to: VectorLike) -> Part | Compound:
        "Returns: A geometrical part, a connector markup part, and a numpy array of connector collision detection positions"
        pass

    def color_and_label(self, geom: Compound, male: bool):
        # note: removing connector_def.
        assert len(geom.children) == 2, "Expected a two children for the geometry."
        assert geom.children[1].volume < geom.children[0].volume, (
            "Expected the connector to be smaller than the geometry."
        )
        assert geom.children[1].volume < 0.2, (
            "Expected the connector to be smaller than 0.2."
        )  # sanity check - connector defs should be tiny. sphere with R=0.3 has volume 0.113
        geom.children[0].color = Color(0.5, 0.5, 0.5, 0.8)
        geom.children[1].color = Color(1, 1, 0, 0.5)

        base_name = self.get_name(self.in_sim_id, male)
        # to indicate typing to the model
        geom.children[0].label = base_name + "@connector"
        geom.children[1].label = base_name + "@connector_def"
        geom.label = base_name + "_compound"

        # util to print connector collision position
        print(
            f"{base_name}_connector_collision_detection_position",
            geom.children[1].center(CenterOf.BOUNDING_BOX),
        )

        return geom

    @staticmethod
    def from_name(name: str) -> "Connector":
        """Get the connector from a name."""
        if "@" in name:
            assert name.endswith("@connector")
            name = name[: -len("@connector")]  # remove part type
        model_id, in_sim_id, male_or_female = name.split("_")
        assert male_or_female in ["male", "female"], (
            "Name should end with male or female."
        )
        assert model_id in ["europlug", "xt60", "round_laptop"], (
            "Name should start with europlug, xt60, or round_laptop."
        )
        from repairs_components.geometry.connectors.models.europlug import Europlug

        # from repairs_components.geometry.connectors.models.xt60 import XT60
        # from repairs_components.geometry.connectors.models.round_laptop import RoundLaptop
        if model_id == "europlug":
            return Europlug(int(in_sim_id))
        # elif model_id == "xt60": #TODO
        #     return XT60(int(in_sim_id), male_or_female == "male")
        # elif model_id == "round_laptop":
        #     return RoundLaptop(int(in_sim_id), male_or_female == "male")
        else:
            raise ValueError(f"Unknown connector name: {name[0]}")

    @staticmethod
    def save_path_from_name(base_dir: Path, name: str, suffix: str = "xml"):
        """Get the connector from a name."""
        if "@" in name:
            assert name.endswith("@connector")
            name = name[: -len("@connector")]  # remove part type
        connector_type, in_sim_id, male_or_female = name.split("_", 2)
        assert len(name.split("_", 2)) == 3, (
            "Name should be of the form <connector_name>_<in_sim_id>_<male_or_female>."
        )
        assert male_or_female in ["male", "female"], (
            "Name should end with male or female."
        )
        assert connector_type.upper() in ConnectorsEnum.__members__.keys(), (
            "Name should start with a valid connector type. Got: " + connector_type
        )  # note `upper()` because enum is uppercase.
        assert in_sim_id.isdigit(), "Name should have an integer as the in_sim_id."
        return (
            base_dir
            / "shared"
            / "connectors"
            / Path(f"{connector_type}_{in_sim_id}_{male_or_female}.{suffix}")
        )

    def get_path(self, base_dir: Path, male: bool) -> Path:
        return self.save_path_from_name(base_dir, self.name(0, male))


def check_connections(
    male_connector_positions: torch.Tensor,
    female_connector_positions: torch.Tensor,
    connection_threshold: float = 2.5,
) -> torch.Tensor:
    """
    Find all male-female connector pairs that are spatially close.

    Computes pairwise distances between batched 3D positions of male and female connectors.
    A connection is valid if the distance is less than `connection_threshold`.

    Parameters
    ----------
    male_connector_positions : torch.Tensor
        Tensor of shape [M, 3] containing male connector positions.
    female_connector_positions : torch.Tensor
        Tensor of shape [F, 3] containing female connector positions.
    connection_threshold : float, default=2.5
        Max distance allowed for a valid connection.

    Returns
    -------
    connections : torch.Tensor
        Tensor of shape [N, 2] where N is the number of valid connections.
        Each row contains [male_idx, female_idx] for a valid connection.
    """
    if male_connector_positions.numel() == 0 or female_connector_positions.numel() == 0:
        return torch.empty(
            (0, 2), dtype=torch.long, device=male_connector_positions.device
        )

    assert (
        male_connector_positions.ndim == 2 and male_connector_positions.shape[1] == 3
    ), (
        f"Expected male_connector_positions to be [M, 3], got {male_connector_positions.shape}"
    )
    assert (
        female_connector_positions.ndim == 2
        and female_connector_positions.shape[1] == 3
    ), (
        f"Expected female_connector_positions to be [F, 3], got {female_connector_positions.shape}"
    )

    # Compute pairwise distances [M, F]
    D = torch.cdist(
        male_connector_positions.unsqueeze(0),
        female_connector_positions.unsqueeze(0),
        p=2,
    ).squeeze(0)

    # Find connections below threshold
    mask = D < connection_threshold
    indices = mask.nonzero(as_tuple=False)  # [N, 2] with (m_idx, f_idx)

    return indices


class ConnectorsEnum(IntEnum):
    "Enum to select connectors"

    EUROPLUG = 0
    XT60 = 1
    ROUND_LAPTOP = 2
    POWERPOLE = 3
    USB_TYPE_C = 4
    USB_TYPE_A = 5
    AMP_SUPERSEAL = 6  # nice connector and easy to model
    # ETHERNET = 7  # just maybe.

    # Note: these kinds of enums are better moved away, but for an unscalable version this is fine.
    @staticmethod
    def by_id(id: int, in_sim_id: int) -> Connector:
        # import everything here because of circular dependencies
        # from repairs_components.geometry.connectors.models.xt60 import XT60

        from repairs_components.geometry.connectors.models.europlug import Europlug
        from repairs_components.geometry.connectors.models.powerpole import Powerpole

        enum_ = ConnectorsEnum(id)
        if enum_ == ConnectorsEnum.EUROPLUG:
            return Europlug(in_sim_id)
        if enum_ == ConnectorsEnum.XT60:
            raise NotImplementedError
        elif enum_ == ConnectorsEnum.ROUND_LAPTOP:
            raise NotImplementedError
        elif enum_ == ConnectorsEnum.POWERPOLE:
            return Powerpole(in_sim_id)
        elif enum_ == ConnectorsEnum.USB_TYPE_C:
            raise NotImplementedError
        elif enum_ == ConnectorsEnum.USB_TYPE_A:
            raise NotImplementedError
        elif enum_ == ConnectorsEnum.AMP_SUPERSEAL:
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown connector id: {id}")
