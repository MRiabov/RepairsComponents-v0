# Components Reference

RepairsComponents provides several types of components for building repair and maintenance simulations. This guide provides an overview of each component type and how to use them.

## Component Base Class

All components inherit from the base `Component` class, which provides common functionality:

```python
from repairs_components import Component

class MyComponent(Component):
    def __init__(self, name=None):
        super().__init__(name=name)
        
    def attach_to_model(self, model, data):
        """Attach the component to a MuJoCo model."""
        super().attach_to_model(model, data)
        # Custom attachment logic here
        
    def step(self):
        """Update the component state for the current timestep."""
        super().step()
        # Custom update logic here
```

## Fasteners

### Screw

A screw that can be fastened and released through rotation.

```python
from repairs_components import Screw

# Create a screw with custom parameters
screw = Screw(
    thread_pitch=0.5,  # mm per rotation
    length=10.0,       # mm
    diameter=3.0,      # mm
    head_diameter=6.0, # mm
    head_height=2.0,   # mm
    name="example_screw"
)

# Fasten the screw (positive rotation)
screw.fasten(np.pi/2)  # Rotate 90 degrees

# Release the screw (negative rotation)
screw.release(np.pi/4)  # Rotate back 45 degrees

# Check if the screw is fully fastened
if screw.is_fully_fastened:
    print("Screw is fully fastened!")
```

## Sockets

### BasicSocket

A simple socket that can connect and disconnect.

```python
from repairs_components import BasicSocket

# Create a basic socket
socket = BasicSocket(
    size=8.0,         # mm
    depth=10.0,       # mm
    wall_thickness=1.0,# mm
    name="example_socket"
)

# Connect the socket
if socket.connect(force=5.0):  # 5 Newtons of force
    print("Socket connected!")


# Disconnect the socket
if socket.disconnect():
    print("Socket disconnected!")
```

### LockingSocket

A socket with a locking mechanism that requires a release action.

```python
from repairs_components import LockingSocket

# Create a locking socket
locking_socket = LockingSocket(
    size=10.0,
    requires_release=True,
    release_force=5.0,  # Force needed to release
    name="locking_socket"
)

# Try to disconnect (will fail without releasing)
if not locking_socket.disconnect():
    print("Cannot disconnect: release not activated")

# Activate release and disconnect
locking_socket.activate_release(force=6.0)  # Must be >= release_force
if locking_socket.disconnect():
    print("Successfully disconnected")
```

## Controls

### Button

A pressable button that can trigger actions.

```python
from repairs_components import Button

def on_button_press(state):
    print(f"Button {'pressed' if state else 'released'}")

# Create a button with a callback
button = Button(
    on_press=on_button_press,
    press_force=2.0,  # Newtons
    initial_state=False,
    size=10.0,  # mm
    name="example_button"
)

# Press the button
if button.press(force=2.5):
    print("Button press successful!")

# Check button state
if button.is_pressed:
    print("Button is pressed")
```

## Creating Custom Components

You can create custom components by extending the base `Component` class:

```python
from repairs_components import Component
import numpy as np

class CustomComponent(Component):
    def __init__(self, custom_param=1.0, name=None):
        super().__init__(name=name)
        self.custom_param = custom_param
        self._state = 0.0
        
    def attach_to_model(self, model, data):
        super().attach_to_model(model, data)
        # Initialize any MuJoCo-specific elements here
        
    def step(self):
        super().step()
        # Update component state each timestep
        self._state = np.sin(self._sim_time * 2 * np.pi * 0.1)  # Example: oscillating state
        
    def custom_method(self, value):
        """Example custom method."""
        self._state += value
        return self._state
```

## Component Lifecycle

1. **Initialization**: Create the component with desired parameters.
2. **Attachment**: Attach to a MuJoCo model using `attach_to_model(model, data)`.
3. **Simulation**: In each simulation step, call the component's `step()` method.
4. **Interaction**: Call the component's methods to interact with it.
5. **Cleanup**: When done, the component will be garbage collected automatically.

## Best Practices

1. **Naming**: Always provide meaningful names to components for debugging.
2. **Error Handling**: Check return values from component methods.
3. **Performance**: Minimize computation in the `step()` method for better performance.
4. **State Management**: Use the component's state methods to save/load states if needed.
