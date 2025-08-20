"""Basic example demonstrating the use of repair components."""

import genesis as gs

from repairs_components.geometry.electrical.consumers.led import Led

# from repairs_components.logic.electronics.connector import Connector
from repairs_components.logic.electronics.simulator import simulate_circuit
from repairs_components.logic.electronics.voltage_source import VoltageSource
from repairs_components.logic.electronics.wire import Wire

# Add the parent directory (project root) to sys.path so src.geometry can be imported
# sys.path.append(str(Path(__file__).parent.parent))

# for AI: atm I'm prototyping electronics circuit logic, so we are connecting round laptop sockets, and the male socket accepts power, and the male socket also powers the lightbulb from the accepted voltage.

gs.init(backend=gs.cpu, logging_level="debug")
scene = gs.Scene(
    show_viewer=False,
    renderer=gs.renderers.Rasterizer(),
)
# socket_female = get_socket_by_type("round_laptop_female")
# socket_male = get_socket_by_type("round_laptop_male")
socket_male = Led(name="l1", size=(1, 1, 1), pos=(0, 0, 0), gs_scene=scene)


socket_female_elec = VoltageSource(10, "v1")
socket_male_elec = Wire("w1")

socket_female_elec.connect(socket_male_elec)
l1 = Lightbulb(10, "l1")
socket_female_elec.connect(l1)
socket_male_elec.connect(l1)
results = simulate_circuit([socket_female_elec, socket_male_elec, l1])

lightbulb_bright = any(
    x["type"] == "lightbulb" and x["brightness"] > 0 for x in results
)
print("lightbulb is bright", lightbulb_bright)
assert lightbulb_bright
