# Controls

## Button

A pressable button that can trigger actions.

```{eval-rst}
.. autoclass:: repairs_components.controls.Button
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
```

### Initialization

```python
def __init__(
    self,
    on_press: Optional[Callable[[bool], None]] = None,
    initial_state: bool = False,
    press_force: float = 1.0,
    size: float = 10.0,
    name: Optional[str] = None
)
```

**Parameters:**
- `on_press`: Optional callback function that takes a boolean parameter (pressed state)
- `initial_state`: Initial pressed state of the button
- `press_force`: Force required to press the button in Newtons
- `size`: Diameter of the button in mm
- `name`: Optional name for the button

### Methods

#### `press`

```python
def press(self, force: float = 1.0) -> bool
```

Attempt to press the button.

**Parameters:**
- `force`: The force applied to the button in Newtons

**Returns:**
- `True` if the button state changed, `False` otherwise

#### `release`

```python
def release(self) -> bool
```

Release the button if it's pressed.

**Returns:**
- `True` if the button was released, `False` if it was already released

### Properties

#### `is_pressed`

```python
@property
def is_pressed(self) -> bool
```

Check if the button is currently pressed.

#### `press_force`

```python
@property
def press_force(self) -> float
```

Get the force required to press the button in Newtons.

#### `size`

```python
@property
def size(self) -> float
```

Get the diameter of the button in mm.

### Example

```python
from repairs_components import Button

def on_button_change(state):
    if state:
        print("Button pressed!")
    else:
        print("Button released!")

# Create a button with a callback
button = Button(
    on_press=on_button_change,
    press_force=2.0,  # 2N to press
    size=12.0,        # 12mm diameter
    name="power_button"
)

# Press the button with 3N of force (more than required)
if button.press(force=3.0):
    print("Button press successful!")

# Check button state
if button.is_pressed:
    print("Button is pressed")

# Release the button
if button.release():
    print("Button released!")
```

## ToggleSwitch

A switch that can be toggled between two states.

```{eval-rst}
.. autoclass:: repairs_components.controls.ToggleSwitch
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
```

### Initialization

```python
def __init__(
    self,
    on_toggle: Optional[Callable[[bool], None]] = None,
    initial_state: bool = False,
    toggle_force: float = 1.0,
    size: float = 10.0,
    name: Optional[str] = None
)
```

**Parameters:**
- `on_toggle`: Optional callback function that takes a boolean parameter (new state)
- `initial_state`: Initial state of the switch
- `toggle_force`: Force required to toggle the switch in Newtons
- `size`: Size of the switch in mm
- `name`: Optional name for the switch

### Example

```python
from repairs_components import ToggleSwitch

# Create a toggle switch
toggle = ToggleSwitch(
    on_toggle=lambda state: print(f"Switch {'on' if state else 'off'}"),
    toggle_force=1.5,  # 1.5N to toggle
    size=15.0,         # 15mm size
    name="power_switch"
)

# Toggle the switch
toggle.toggle(force=2.0)  # Output: "Switch on"
toggle.toggle(force=2.0)  # Output: "Switch off"
```

## Exceptions

### ControlError

Base class for all control-related exceptions.

```{eval-rst}
.. autoexception:: repairs_components.controls.ControlError
   :members:
```

### ButtonError

Raised when there is an error with a button operation.

```{eval-rst}
.. autoexception:: repairs_components.controls.ButtonError
   :members:
```

## Constants

### MIN_BUTTON_SIZE

Minimum allowed button size in mm.

### MAX_PRESS_FORCE

Maximum allowed press force in Newtons.
