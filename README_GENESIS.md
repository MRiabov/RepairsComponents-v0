# Repairs Components for Genesis

This is a port of the Repairs Components library to work with the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) simulator.

## Installation

1. First, install Genesis following the [official installation instructions](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html).

2. Install this package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

### Basic Example

Here's how to use the Genesis version of the components:

```python
import genesis as gs
from repairs_components.genesis_component import GenesisComponent

# Initialize Genesis
gs.init(backend="cpu", logging_level='warning')

# Create a scene
scene = gs.Scene(show_viewer=True)

# Add a plane as the ground
plane = scene.add_entity(gs.morphs.Plane())

# Create and add your repair component
# component = YourComponent()
# component.attach_to_scene(scene)

# Build the scene
scene.build()

# Run the simulation
for _ in range(1000):
    scene.step()
```

### Running the Example

Run the example script to see a simple simulation:

```bash
python examples/genesis_button_demo.py
```

## Migration from MuJoCo

If you're migrating from MuJoCo to Genesis, here are the key changes:

1. **Initialization**:
   - MuJoCo: `mujoco.MjModel` and `mujoco.MjData`
   - Genesis: `gs.Scene` and `gs.morphs`

2. **Simulation Loop**:
   - MuJoCo: `mujoco.mj_step(model, data)`
   - Genesis: `scene.step()`

3. **Rendering**:
   - MuJoCo: Uses `mujoco.viewer`
   - Genesis: Built into `gs.Scene(show_viewer=True)`

## Creating Custom Components

To create a custom component, inherit from `GenesisComponent`:

```python
from repairs_components.genesis_component import GenesisComponent

class YourComponent(GenesisComponent):
    def __init__(self, name: str = ""):
        super().__init__(name)
        # Your initialization code here
    
    def to_mjcf(self) -> str:
        # Return MJCF XML string for this component
        return """
        <mujoco>
            <!-- Your MJCF definition -->
        </mujoco>
        """
    
    def attach_to_scene(self, scene):
        super().attach_to_scene(scene)
        # Your code to add entities to the scene
        pass
    
    def step(self):
        # Your per-step simulation code
        pass
```

## License

This project is licensed under the same license as the original Repairs Components library.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
