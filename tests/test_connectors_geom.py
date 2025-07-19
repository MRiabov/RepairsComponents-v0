from build123d import CenterOf, Compound, Color, Location
import pytest
from repairs_components.geometry.connectors.connectors import ConnectorsEnum
from repairs_components.geometry.connectors.models.europlug import Europlug
import numpy as np

from repairs_components.geometry.connectors.models.powerpole import Powerpole
from repairs_components.geometry.connectors.models.usb_c import USB_C
from repairs_components.geometry.connectors.models.usb_type_a import USB_A
from repairs_components.geometry.connectors.connectors import Connector


@pytest.fixture
def all_connectors() -> list[Connector]:
    return [Europlug(0), USB_C(1), USB_A(2), Powerpole(3)]


# TODO: couldn't validate...
# dev note: the logic is that connector def is at the same position as connector.
# ^ however the actual implementation e.g. in WireUp `moves` the connectors. Which worsks, even though I, 5 days later after writing it, don't expect it.
def test_connected_connectors_match(all_connectors: list[Connector]):
    for connector_dataclass in all_connectors:
        bd_geometry_male, connector_pos, bd_geometry_female, _ = (
            connector_dataclass.bd_geometry((0, 0, 0), connected=True)
        )
        # dev note: the logic to make this work could be:
        # bd_geometry_male = bd_geometry_male.move(
        #     Location(bd_geometry_male.location.position)
        # )
        # bd_geometry_female = bd_geometry_female.move(
        #     Location(bd_geometry_female.location.position)
        # )
        # ^ *move()* is not the same as *moved()*, it moves the children, not the parent; move self and maybe it'll work.
        # ... no, as expected, it only duplicated the pos...

        assert np.isclose(
            np.array(
                tuple(bd_geometry_male.children[-1].center(CenterOf.BOUNDING_BOX))
            ),
            np.array(tuple(connector_pos)),
        ).all()
        assert np.isclose(
            np.array(
                tuple(bd_geometry_female.children[-1].center(CenterOf.BOUNDING_BOX))
            ),
            np.array(tuple(connector_pos)),
        ).all()  # consequently connector defs in both male and female should be at the same position.
        ##hmmm. this isn't expected to work either because the connector def is at

        no_connector_def_male = Compound(children=[bd_geometry_male.children[0]])
        no_connector_def_female = Compound(children=[bd_geometry_female.children[0]])
        cleaned_up_intersection_compound = Compound(
            children=[no_connector_def_male, no_connector_def_female]
        )
        # TODO use cleaned up intersect!!!
        cleaned_up_intersection_bool, parts, volume = (
            cleaned_up_intersection_compound.do_children_intersect()
        )
        assert cleaned_up_intersection_bool == False, (
            f"Connectors should never intersect. Volume: {volume}"
        )
        assert parts is None
        assert volume == 0


def test_connector_pos_relative_to_center_match(all_connectors: list[Connector]):
    for connector_dataclass in all_connectors:
        bd_geometry_male = connector_dataclass.bd_geometry_male((0, 0, 0))
        bd_geometry_female = connector_dataclass.bd_geometry_female((0, 0, 0))

        assert np.isclose(
            tuple(bd_geometry_male.children[-1].center(CenterOf.BOUNDING_BOX)),
            connector_dataclass.connector_pos_relative_to_center_male,
        ).all(), (
            f"Male connector pos relative to center doesn't match. It is set as {connector_dataclass.connector_pos_relative_to_center_male}"
            f" but it is {bd_geometry_male.children[-1].center(CenterOf.BOUNDING_BOX)}"
        )
        assert np.isclose(
            tuple(bd_geometry_female.children[-1].center(CenterOf.BOUNDING_BOX)),
            connector_dataclass.connector_pos_relative_to_center_female,
        ).all(), (
            f"Female connector pos relative to center doesn't match. It is set as {connector_dataclass.connector_pos_relative_to_center_female}"
            f" but it is {bd_geometry_female.children[-1].center(CenterOf.BOUNDING_BOX)}"
        )
        # ^ now, this is expected to work
        assert (
            connector_dataclass.connector_pos_relative_to_center_male[1:]
            == np.array([0, 0])
        ).all(), "expected y and z of connector defs to be 0"
        assert (
            connector_dataclass.connector_pos_relative_to_center_female[1:]
            == np.array([0, 0])
        ).all(), "expected y and z of connector defs to be 0"


def test_naming_and_colors(all_connectors: list[Connector]):
    for connector_dataclass in all_connectors:
        bd_geometry_male, connector_pos, bd_geometry_female, _ = (
            connector_dataclass.bd_geometry((0, 0, 0), connected=True)
        )
        # has two children only
        assert len(bd_geometry_male.children) == 2
        assert len(bd_geometry_female.children) == 2
        # connector def names
        assert bd_geometry_male.children[-1].label.endswith("@connector_def")
        assert bd_geometry_female.children[-1].label.endswith("@connector_def")
        # connector names
        assert bd_geometry_male.children[0].label.endswith("male@connector")
        assert bd_geometry_female.children[0].label.endswith("female@connector")
        # connector colors male and female
        assert np.isclose(
            tuple(bd_geometry_male.children[0].color), tuple(Color(0.5, 0.5, 0.5, 0.8))
        ).all()
        assert np.isclose(
            tuple(bd_geometry_male.children[1].color), tuple(Color(1, 1, 0, 0.5))
        ).all()
        assert np.isclose(
            tuple(bd_geometry_female.children[0].color),
            tuple(Color(0.5, 0.5, 0.5, 0.8)),
        ).all()
        assert np.isclose(
            tuple(bd_geometry_female.children[1].color), tuple(Color(1, 1, 0, 0.5))
        ).all()
