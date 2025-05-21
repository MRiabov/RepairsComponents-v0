from genesis import gs


def get_socket_by_type(connector_type: str):
    assert connector_type in [
        "iec13",
        "XT60",
        "round_laptop_female",
        "round_laptop_male",
    ]

    return gs.morphs.Mesh(
        file="geom_exports/electronics/connectors/" + connector_type + ".gltf"
    )
