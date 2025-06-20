from enum import IntEnum
import sys
import os

from build123d import Compound, Part, VectorLike, Color
from genesis import gs
from repairs_components.logic.electronics.component import (
    ElectricalComponent,
    ElectricalComponentsEnum,
)
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Mapping


class Connector(ElectricalComponent):
    def propagate(self, voltage: float, current: float) -> tuple[float, float]:
        return voltage, current  # pass through.

    @property
    def component_type(self) -> int:
        return ElectricalComponentsEnum.CONNECTOR.value

    def get_mjcf(
        self, connector_position: np.ndarray | None = None, density: float = 1000
    ):
        "This defines a generic mjcf for a connector with a connector position."
        if connector_position is None:
            connector_position = self.connector_pos_relative_to_center
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
        self, moved_to_male: VectorLike, moved_to_female: VectorLike
    ) -> tuple[Part | Compound, Part, np.ndarray, Part | Compound, Part, np.ndarray]:
        """
        return build123d geometry of containers, colored as the connector color and with connector metadata.
        """
        geom_male, male_connector_def, male_connector_collision_detection_position = (
            self.bd_geometry_male(moved_to_male)
        )
        (
            geom_female,
            female_connector_def,
            female_connector_collision_detection_position,
        ) = self.bd_geometry_female(moved_to_female)

        geom_male.color = Color(0.5, 0.5, 0.5, 0.8)
        geom_female.color = Color(0.5, 0.5, 0.5, 0.8)

        male_connector_def.color = Color(1, 1, 0, 0.5)  # yellow
        female_connector_def.color = Color(1, 1, 0, 0.5)

        return (
            geom_male,
            male_connector_def,
            male_connector_collision_detection_position,
            geom_female,
            female_connector_def,
            female_connector_collision_detection_position,
        )

    @abstractmethod
    def bd_geometry_male(
        self, moved_to: VectorLike
    ) -> tuple[Part | Compound, Part, np.ndarray]:
        "Returns: A geometrical part, a connector markup part, and a numpy array of connector collision detection positions"
        pass

    @abstractmethod
    def bd_geometry_female(
        self, moved_to: VectorLike
    ) -> tuple[Part | Compound, Part, np.ndarray]:  # type:ignore
        "Returns: A geometrical part, a connector markup part, and a numpy array of connector collision detection positions"
        pass

    def color_and_label(self, geom: Part, connector_def: Part):
        geom.color = Color(0.5, 0.5, 0.5, 0.8)
        connector_def.color = Color(1, 1, 0, 0.5)

        # to indicate typing to the model
        geom.label = self.name + "@solid"
        connector_def.label = self.name + "@connector"

        return geom, connector_def


def check_connections(
    male_connectors: Mapping[str, torch.Tensor | np.ndarray],
    female_connectors: Mapping[str, torch.Tensor | np.ndarray],
    connection_threshold: float = 2.5,
) -> torch.Tensor:
    """
    Identify feasible male–female connector pairings based on Euclidean proximity.

    Let X_m ∈ ℝ^{M×3} and X_f ∈ ℝ^{F×3} be the arrays of 3D coordinates for M
    male and F female connector endpoints, respectively. We define the pairwise
    distance matrix D ∈ ℝ^{M×F} by

        D_{i,j} = || X_m[i, :] − X_f[j, :] ||_2

    A binary adjacency mask A ∈ {0,1}^{M×F} is then obtained by

        A_{i,j} = 1   if   D_{i,j} < τ
                = 0   otherwise

    where τ is the scalar `connection_threshold`. This function returns the list
    of all (key_m, key_f) pairs for which A_{i,j} = 1, where key_m and key_f are
    the dictionary keys from male_connectors and female_connectors respectively.

    Parameters
    ----------
    male_connectors : Mapping[str, torch.Tensor | np.ndarray]
        Dictionary mapping male connector IDs to their 3D coordinates.
    female_connectors : Mapping[str, torch.Tensor | np.ndarray]
        Dictionary mapping female connector IDs to their 3D coordinates.
    connection_threshold : float, default=2.5
        Maximum allowable Euclidean distance for a valid connection.

    Returns
    -------
    connections : torch.Tensor
        Tensor of shape [num_pairs, 3] with (batch, male_idx, female_idx)
    """
    male_keys = list(male_connectors.keys())
    female_keys = list(female_connectors.keys())
    if not male_keys or not female_keys:
        # determine batch size from first tensor
        first = next(iter(male_connectors.values()))
        batch = first.shape[0] if hasattr(first, "shape") else 1
        return torch.empty((batch, 0, 3), dtype=torch.long)
    # GPU-accelerated torch computation: compute batch distances
    # detect device
    first_val = next(iter(male_connectors.values()))
    device = (
        first_val.device if isinstance(first_val, torch.Tensor) else torch.device("cpu")
    )

    def to_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x, device=device)

    male_vals = torch.stack([to_tensor(male_connectors[k]) for k in male_keys], dim=1)
    female_vals = torch.stack(
        [to_tensor(female_connectors[k]) for k in female_keys], dim=1
    )
    # male_vals: [B, M, 3], female_vals: [B, F, 3]
    D = torch.linalg.norm(male_vals[:, :, None, :] - female_vals[:, None, :, :], dim=-1)
    mask = D < connection_threshold
    # return indices tensor of shape [num_pairs, 3] with (batch, male_idx, female_idx)
    return mask.nonzero(as_tuple=False)


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
        enum_= ConnectorsEnum(id)
        if enum_ == ConnectorsEnum.EUROPLUG:
            return Europlug(name)
        if enum_ == ConnectorsEnum.XT60:
            raise NotImplementedError
        elif enum_ == ConnectorsEnum.ROUND_LAPTOP:
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown connector id: {id}")
