import sys
import os

from build123d import Compound, Part, VectorLike, Color
from genesis import gs

import src.geometry.connectors.models as models

from src.geometry.base import Component
from src.geometry.connectors.models import europlug, round
from abc import ABC, abstractmethod


def get_socket_mesh_by_type(connector_type: str):
    assert connector_type in [
        "iec13",
        "XT60",
        "round_laptop_female",
        "round_laptop_male",
    ]

    return gs.morphs.Mesh(
        file="geom_exports/electronics/connectors/" + connector_type + ".gltf"
    )


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


class Connector(Component):
    name = "connector"  # abstract

    def bd_geometry(
        self, moved_to_male: VectorLike, moved_to_female: VectorLike
    ) -> tuple[Part | Compound, Part, Part | Compound, Part]:  # type:ignore
        """
        return build123d geometry of containers, colored as the connector color and with connector metadata.
        """
        geom_male, male_connector_def = self.bd_geometry_male(moved_to_male)
        geom_female, female_connector_def = self.bd_geometry_female(moved_to_female)

        geom_male.color = Color(0.5, 0.5, 0.5, 0.8)
        geom_female.color = Color(0.5, 0.5, 0.5, 0.8)

        male_connector_def.color = Color(1, 1, 0, 0.5)  # yellow
        female_connector_def.color = Color(1, 1, 0, 0.5)

        return geom_male, male_connector_def, geom_female, female_connector_def

    @abstractmethod
    def bd_geometry_male(self, moved_to: VectorLike) -> tuple[Part | Compound, Part]:  # type:ignore
        pass

    @abstractmethod
    def bd_geometry_female(self, moved_to: VectorLike) -> tuple[Part | Compound, Part]:  # type:ignore
        pass

    def color_and_label(self, geom: Part, connector_def: Part):
        geom.color = Color(0.5, 0.5, 0.5, 0.8)
        connector_def.color = Color(1, 1, 0, 0.5)

        # to indicate typing to the model
        geom.label = self.name + "@solid"
        connector_def.label = self.name + "@connector"

        return geom, connector_def
