from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PartType(str, Enum):
    SOLID = "solid"
    FIXED_SOLID = "fixed_solid"
    CONNECTOR = "connector"
    TERMINAL_DEF = "terminal_def"
    FASTENER = "fastener"
    BUTTON = "button"
    LED = "led"
    SWITCH = "switch"
    LIQUID = "liquid"


class ConnectorSex(str, Enum):
    NONE = ""
    MALE = "male"
    FEMALE = "female"


@dataclass(frozen=True)
class ParsedPartLabel:
    full: str
    name: str  # without the "@type" suffix
    type: PartType
    connector_sex: ConnectorSex = ConnectorSex.NONE


_VALID_TYPES = {t.value: t for t in PartType}


def parse_part_label(label: str) -> ParsedPartLabel:
    """Parse a part label of the form "<name>@<type>" into a structured representation.

    Contract:
    - label must contain a single '@' delimiter separating name and type
    - type must be one of PartType values (lowercase string)
    - connectors must encode sex in name via "_male" or "_female"
    """
    assert label is not None and "@" in label, f"Label must contain '@': {label}"
    name, type_str = label.split("@", 1)
    type_str = type_str.lower()
    assert type_str in _VALID_TYPES, f"Unknown part type: {type_str} in {label}"

    ptype = _VALID_TYPES[type_str]

    sex = ConnectorSex.NONE
    if ptype == PartType.CONNECTOR:
        if name.endswith("_male"):
            sex = ConnectorSex.MALE
        elif name.endswith("_female"):
            sex = ConnectorSex.FEMALE
        else:
            # Keep strict: all connectors must explicitly declare sex
            assert False, f"Connector name must end with _male or _female: {label}"

    return ParsedPartLabel(full=label, name=name, type=ptype, connector_sex=sex)
