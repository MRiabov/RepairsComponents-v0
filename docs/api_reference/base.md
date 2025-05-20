# Base Classes

## Component

Base class for all repair components.

```{eval-rst}
.. autoclass:: repairs_components.base.Component
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
```

### Methods

#### `__init__`

```python
def __init__(self, name: Optional[str] = None)
```

Initialize the component.

**Parameters:**
- `name`: Optional name for the component. If not provided, a unique name will be generated.

#### `attach_to_model`

```python
def attach_to_model(self, model: Any, data: Any) -> None
```

Attach the component to a MuJoCo model.

**Parameters:**
- `model`: The MuJoCo model to attach to
- `data`: The MuJoCo data object

#### `step`

```python
def step(self) -> None
```

Update the component state for the current timestep.

#### `reset`

```python
def reset(self) -> None
```

Reset the component to its initial state.

### Properties

#### `name`

```python
@property
def name(self) -> str
```

Get the name of the component.

#### `is_attached`

```python
@property
def is_attached(self) -> bool
```

Check if the component is attached to a model.

### Example

```python
from repairs_components import Component
#import mujoco #TODO replace mujoco with Genesis!

class CustomComponent(Component):
    def __init__(self, param=1.0, name=None):
        super().__init__(name=name)
        self.param = param
        self._state = 0.0
        
    def attach_to_model(self, model, data):
        super().attach_to_model(model, data)
        # Initialize MuJoCo-specific elements here
        
    def step(self):
        super().step()
        # Update component state
        self._state += self.param * 0.01  # Example: integrate over time

# Usage
model = mujoco.MjModel.from_xml_string("<mujoco><worldbody></worldbody></mujoco>")
data = mujoco.MjData(model)
component = CustomComponent(param=2.0, name="my_component")
component.attach_to_model(model, data)

# In simulation loop
for _ in range(100):
    component.step()
```

## Exceptions

### ComponentError

Base class for all component-related exceptions.

```{eval-rst}
.. autoexception:: repairs_components.base.ComponentError
   :members:
```

### AttachmentError

Raised when there is an error attaching a component to a model.

```{eval-rst}
.. autoexception:: repairs_components.base.AttachmentError
   :members:
```

### StateError

Raised when a component is in an invalid state for the requested operation.

```{eval-rst}
.. autoexception:: repairs_components.base.StateError
   :members:
```

## Constants

### DEFAULT_FRICTION

Default friction coefficient for component interactions.

### DEFAULT_DAMPING

Default damping coefficient for component motion.
