# Quickstart Guide

This guide will walk you through creating a simple simulation using RepairsComponents.

## Basic Usage

### Creating Components

```python
#import mujoco #TODO replace mujoco with Genesis!
from repairs_components import Screw, BasicSocket, Button

# Create a simple MuJoCo model
model = mujoco.MjModel.from_xml_string("""
<mujoco>
    <option timestep="0.01"/>
    <worldbody>
        <light name="light" pos="0 0 4"/>
        <camera name="fixed" pos="0 -1 0.5" xyaxes="1 0 0 0 0 1"/>
        <geom name="floor" type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
    </worldbody>
</mujoco>
""")
data = mujoco.MjData(model)

# Create components
screw = Screw(thread_pitch=0.5, length=10.0, name="test_screw")
button = Button(press_force=2.0, name="test_button")
socket = BasicSocket(size=8.0, name="test_socket")

# Attach components to the model
for component in [screw, button, socket]:
    component.attach_to_model(model, data)
```

### Using Components

```python
# Fasten the screw
screw.fasten(np.pi/2)  # Rotate 90 degrees
print(f"Screw position: {screw._position} mm")

# Press the button
if button.press(force=2.5):
    print("Button pressed!")

# Connect the socket
if socket.connect(force=10.0):
    print("Socket connected!")
```

## Component Interaction

Components can interact with each other through callbacks:

```python
def on_button_press(state):
    if state:
        print("Button pressed - releasing socket")
        socket.disconnect()

# Create a button that releases the socket when pressed
button = Button(
    on_press=on_button_press,
    press_force=2.0,
    name="release_button"
)
```

## Simulation Loop

Here's how to integrate components into a simulation loop:

```python
import time

def simulate(model, data, components, steps=1000):
    """Run a simulation with the given components."""
    for _ in range(steps):
        # Apply forces, update controls, etc.
        
        # Step the simulation
        mujoco.mj_step(model, data)
        
        # Update components
        for component in components:
            component.step()
        
        # Optional: Add a small delay for visualization
        time.sleep(0.01)

# Run the simulation
components = [screw, button, socket]
simulate(model, data, components, steps=1000)
```

## Visualizing with MuJoCo Viewer

For better visualization, you can use the MuJoCo viewer:

```python
from mujoco import viewer

def simulate_with_viewer(model, data, components, steps=1000):
    """Run a simulation with the MuJoCo viewer."""
    with viewer.launch_passive(model, data) as v:
        # Set camera
        v.cam.distance = 3.0
        v.cam.azimuth = 0
        v.cam.elevation = -20
        
        # Main simulation loop
        for _ in range(steps):
            # Apply forces, update controls, etc.
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Update components
            for component in components:
                component.step()
            
            # Sync the viewer
            v.sync()

# Run the simulation with the viewer
simulate_with_viewer(model, data, components, steps=1000)
```

## Next Steps

- Explore the [API Reference](../api_reference/index.md) for detailed documentation of all components
- Check out the [examples](../examples/index.md) for more complex use cases
- Learn how to [create custom components](custom_components.md)
