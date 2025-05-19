# Fasteners

## Screw

A screw that can be fastened and released through rotation.

```{eval-rst}
.. autoclass:: repairs_components.fasteners.Screw
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
```

### Initialization

```python
def __init__(
    self,
    thread_pitch: float = 0.5,
    length: float = 10.0,
    diameter: float = 3.0,
    head_diameter: float = 6.0,
    head_height: float = 2.0,
    friction: float = 0.1,
    name: Optional[str] = None
)
```

**Parameters:**
- `thread_pitch`: Distance between threads in mm
- `length`: Total length of the screw in mm
- `diameter`: Diameter of the screw shaft in mm
- `head_diameter`: Diameter of the screw head in mm
- `head_height`: Height of the screw head in mm
- `friction`: Friction coefficient for the screw
- `name`: Optional name for the screw

### Methods

#### `fasten`

```python
def fasten(self, angle: float) -> float
```

Rotate the screw to fasten it.

**Parameters:**
- `angle`: Rotation angle in radians (positive for fastening, negative for loosening)

**Returns:**
- The actual rotation achieved in radians

#### `release`

```python
def release(self, angle: float) -> float
```

Rotate the screw to release it.

**Parameters:**
- `angle`: Rotation angle in radians (positive for releasing, negative for fastening)

**Returns:**
- The actual rotation achieved in radians

### Properties

#### `position`

```python
@property
def position(self) -> float
```

Get the current position of the screw in mm.

#### `is_fully_fastened`

```python
@property
def is_fully_fastened(self) -> bool
```

Check if the screw is fully fastened.

#### `is_fully_released`

```python
@property
def is_fully_released(self) -> bool
```

Check if the screw is fully released.

### Example

```python
from repairs_components import Screw
import numpy as np

# Create a screw
screw = Screw(
    thread_pitch=0.5,  # 0.5mm per rotation
    length=10.0,       # 10mm long
    diameter=3.0,      # 3mm shaft diameter
    head_diameter=6.0, # 6mm head diameter
    head_height=2.0,   # 2mm head height
    name="example_screw"
)

# Fasten the screw by 90 degrees
screw.fasten(np.pi/2)

print(f"Screw position: {screw.position:.2f}mm")

# Output: Screw position: 0.13mm (0.5mm * 90/360)

# Check if fully fastened
if screw.is_fully_fastened:
    print("Screw is fully fastened!")
```

## Exceptions

### ScrewError

Base class for all screw-related exceptions.

```{eval-rst}
.. autoexception:: repairs_components.fasteners.ScrewError
   :members:
```

### ScrewJammedError

Raised when a screw becomes jammed during rotation.

```{eval-rst}
.. autoexception:: repairs_components.fasteners.ScrewJammedError
   :members:
```

## Constants

### DEFAULT_THREAD_PITCH

Default thread pitch in mm.

### MAX_THREAD_PITCH

Maximum allowed thread pitch in mm.

### MIN_SCREW_LENGTH

Minimum allowed screw length in mm.
