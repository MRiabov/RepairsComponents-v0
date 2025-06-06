# Repairs Components

A modular library of repair components for reinforcement learning environments, built on MuJoCo physics with support for the Genesis simulator.

## Overview

Repairs Components is an open-source library designed to provide a comprehensive set of reusable, physics-based repair components for building realistic repair and maintenance simulations. The library is specifically tailored for use with reinforcement learning environments, particularly those built on the MuJoCo physics engine. By leveraging the power of MuJoCo, Repairs Components enables the creation of highly realistic and interactive simulations, allowing for more effective training and testing of reinforcement learning models.

The library is built around a modular architecture, allowing users to easily integrate and combine different repair components to create complex scenarios. The components themselves are designed to be highly customizable, with a wide range of parameters and settings that can be adjusted to suit specific use cases.

## Features

- **Fasteners**: 
  - Screws with realistic threading and fastening mechanics
  - Configurable thread pitch, length, and head dimensions
  - Realistic torque and force simulation

- **Sockets**: 
  - Basic sockets with simple plug/unplug mechanics
  - Locking sockets with release mechanisms
  - Configurable size, depth, and wall thickness

## Installation

### Prerequisites

- Python 3.8+
- MuJoCo 2.3.0+ (for MuJoCo backend)
- [Genesis Simulator](https://github.com/Genesis-Embodied-AI/Genesis) (for Genesis backend)
- pip

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/RepairsComponents-v0.git
cd RepairsComponents-v0

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Or for minimal installation
# pip install -e .
```

## Backends

### MuJoCo Backend

The original implementation uses MuJoCo as the physics engine:

```python
#import mujoco #TODO replace mujoco with Genesis!
from repairs_components import Screw, BasicSocket, LockingSocket

# Create a simple MuJoCo model
```

### Genesis Backend

For improved performance, you can use the Genesis backend:

```python
import genesis as gs
from repairs_components.genesis_component import GenesisComponent

# Initialize Genesis
gs.init(backend="cpu")

# Create a scene and add components
```

See [README_GENESIS.md](README_GENESIS.md) for more details on using the Genesis backend.

## Quick Start (MuJoCo)
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

# Create and attach components
screw = Screw(thread_pitch=0.5, length=10.0, name="test_screw")
screw.attach_to_model(model, data)

# Fasten the screw (90 degree rotation)
screw.fasten(np.pi/2)
print(f"Screw position: {screw._position} mm")

# Create and use a locking socket
locking_socket = LockingSocket(size=10.0, requires_release=True, name="test_socket")
locking_socket.attach_to_model(model, data)

# Try to disconnect (will fail without releasing)
if not locking_socket.disconnect():
    print("Cannot disconnect: release not activated")

# Activate release and disconnect
locking_socket.activate_release(force=6.0)  # Must be >= release_force
if locking_socket.disconnect():
    print("Successfully disconnected")
```

## Project Structure

```
RepairsComponents-v0/
├── src/
│   └── repairs_components/
│       ├── __init__.py       # Package exports
│       ├── base.py           # Base component class
│       ├── fasteners.py      # Screw and other fastener implementations
│       └── sockets.py        # Socket implementations
├── tests/                    # Unit tests
├── examples/                 # Example scripts
├── docs/                     # Documentation
├── pyproject.toml            # Build system configuration
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Development dependencies
└── README.md                 # This file
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=repairs_components
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -r requirements-dev.txt

# Build the docs
cd docs
make html

# Open the documentation
open _build/html/index.html
```

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- mypy for static type checking
- pylint for code quality

Run the linters and formatters:

```bash
# Format code
black .

# Sort imports
isort .

# Check types
mypy .
# Lint code
pylint src/
```

## Documentation

Detailed API documentation is available in the `docs` directory. Build it locally or view it online at [Documentation Link].

## License

MIT

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Acknowledgements

- Built on [MuJoCo](https://mujoco.org/)
- Inspired by real-world repair and maintenance tasks
