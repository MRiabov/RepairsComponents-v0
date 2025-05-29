import sys
import os

from build123d import Compound, Part, VectorLike, Color
from genesis import gs
from src.logic.electronics.component import ElectricalComponent
import numpy as np

import src.geometry.connectors.models as models

from src.geometry.connectors.models import europlug, round
from abc import ABC, abstractmethod


def get_socket_mesh_by_type(connector_type: str):
    assert connector_type in [
        "europlug",
        "XT60",
        "round_laptop_female",
        "round_laptop_male",
    ]

    return gs.morphs.MJCF(
        file="geom_exports/electronics/connectors/" + connector_type + ".xml"
    )  # note: these mjcf files are simply meshes + connection site.

    # site = np.array([0, 0, 0])
    # return gs.morphs.Mesh(
    #     file="geom_exports/electronics/connectors/" + connector_type + ".gltf"
    # )


def get_socket_bd_geometry_by_type(connector_type: str):
    assert connector_type in [
        "iec13_male",
        "iec13_female",
        "XT60_male",
        "XT60_female",
        "round_laptop_female",
        "round_laptop_male",
    ]
    geom: Part | Compound
    match connector_type:
        case "iec13_male":
            geom = iec13iec_plug_male()
        case "iec13_female":
            geom = iec13.iec_plug_female()
        case "XT60_male":
            geom = round.round_plug()
        case "XT60_female":
            geom = round.round_socket()
        case "round_laptop_female":
            geom = round.round_plug()
        case "round_laptop_male":
            geom = round.round_socket()
    return geom


class Connector(ElectricalComponent):
    name = "connector"  # abstract

    def get_mjcf(self, connector_position: np.ndarray, density: float):
        "This defines a generic mjcf for a connector with a connector position."
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
    male_connectors: dict[str, tuple[float, float, float]],
    female_connectors: dict[str, tuple[float, float, float]],
    connection_threshold: float = 2.5,
) -> list[tuple[str, str]]:
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
    male_connectors : Dict[str, Tuple[float, float, float]]
        Dictionary mapping male connector IDs to their 3D coordinates.
    female_connectors : Dict[str, Tuple[float, float, float]]
        Dictionary mapping female connector IDs to their 3D coordinates.
    connection_threshold : float, default=2.5
        Maximum allowable Euclidean distance for a valid connection.

    Returns
    -------
    connections : List[Tuple[str, str]]
        List of (male_key, female_key) pairs where the distance between the
        corresponding connectors is less than connection_threshold.
    """
    # Extract keys and values, maintaining order
    male_keys = list(male_connectors.keys())
    female_keys = list(female_connectors.keys())

    # Convert to numpy arrays for vectorized operations
    X_m = np.array(list(male_connectors.values()))
    X_f = np.array(list(female_connectors.values()))

    # Compute the full M×F distance matrix via broadcasting
    D = np.linalg.norm(X_m[:, None, :] - X_f[None, :, :], axis=2)

    # Build the binary adjacency mask and get indices of connected pairs
    idx_pairs = np.argwhere(D < connection_threshold)

    # Convert indices back to keys
    return [(male_keys[i], female_keys[j]) for i, j in idx_pairs]
