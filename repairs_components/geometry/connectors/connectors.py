from enum import IntEnum
import sys
import os

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
from typing import Mapping


class Connector(ElectricalComponent):
    _connector_def_size = 0.3  # vis only

    def propagate(self, voltage: float, current: float) -> tuple[float, float]:
        return voltage, current  # pass through.

    @property
    def component_type(self) -> int:
        return ElectricalComponentsEnum.CONNECTOR.value

    @property
    def connected_at_angle(self) -> tuple[float, float, float]:
        return (0, 0, 180)

    def get_mjcf(
        self, connector_position: np.ndarray | None = None, density: float = 1000
    ):
        "This defines a generic mjcf for a connector with a connector position."
        if connector_position is None:
            connector_position = self.connector_pos_relative_to_center
        print(
            "warning: exporting electronics as mjcf is deprecated; replace with native genesis controls."
        )
        return f"""<mujoco>
        <asset>
            <mesh name="{self.name}" file="geom_exports/electronics/connectors/{self.name}.gltf"/>
        </asset>
        
        <worldbody>
            <body name="{self.name}">
                <geom name="{self.name}_geom" type="mesh" mesh="{self.name}" density="{density}"/>
                
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
            geom_male = Compound(
                children=[child.rotate(Axis.Z, 180) for child in geom_male.children]
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

        # to indicate typing to the model
        male_or_female = "male" if male else "female"
        geom.children[
            0
        ].label = f"{self.name}_{male_or_female}@connector"  # should be @connector?
        geom.children[1].label = f"{self.name}_{male_or_female}@connector_def"
        # e.g. europlug_{i}_male@connector_def
        geom.label = f"{self.name}_{male_or_female}_compound"

        return geom


def check_connections(
    male_connectors: Mapping[str, torch.Tensor],
    female_connectors: Mapping[str, torch.Tensor],
    connection_threshold: float = 2.5,
) -> list[list[tuple[str, str]]]:
    """
    Find all male-female connector pairs that are spatially close.

    Computes pairwise distances between batched 3D positions of male and female connectors.
    A connection is valid if the distance is less than `connection_threshold`.

    Parameters
    ----------
    male_connectors : Mapping[str, torch.Tensor]
        Dict mapping male connector names to tensors of shape [B, 3].
    female_connectors : Mapping[str, torch.Tensor]
        Dict mapping female connector names to tensors of shape [B, 3].
    connection_threshold : float, default=2.5
        Max distance allowed for a valid connection.

    Returns
    -------
    connections : list[list[tuple[str, str]]]
        List of length B (batch size). Each element is a list of (male_name, female_name)
        pairs that are valid connections in that batch.
    """
    male_keys = list(male_connectors.keys())
    female_keys = list(female_connectors.keys())

    male_vals = torch.stack(list(male_connectors.values()), dim=1)  # [B, M, 3]
    female_vals = torch.stack(
        list(female_connectors.values()), dim=1
    )  # [B, F, 3]

    D = torch.cdist(male_vals, female_vals, p=2)  # [B, M, F]
    mask = D < connection_threshold
    indices = mask.nonzero(as_tuple=False)  # [N, 3] with (batch, m_idx, f_idx)

    result: list[list[tuple[str, str]]] = [[] for _ in range(male_vals.shape[0])]
    for b, m_idx, f_idx in indices.tolist():
        result[b].append((male_keys[m_idx], female_keys[f_idx]))
    return result


class ConnectorsEnum(IntEnum):
    "Enum to select connectors"

    EUROPLUG = 0
    XT60 = 1
    ROUND_LAPTOP = 2

    # Note: these kinds of enums are better moved away, but for an unscalable version this is fine.
    @staticmethod
    def by_id(id: int, name: str) -> Connector:
        # import everything here because of circular dependencies
        # from repairs_components.geometry.connectors.models.xt60 import XT60

        from repairs_components.geometry.connectors.models.europlug import Europlug

        enum_ = ConnectorsEnum(id)
        if enum_ == ConnectorsEnum.EUROPLUG:
            return Europlug(name)
        if enum_ == ConnectorsEnum.XT60:
            raise NotImplementedError
        elif enum_ == ConnectorsEnum.ROUND_LAPTOP:
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown connector id: {id}")
