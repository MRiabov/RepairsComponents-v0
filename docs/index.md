# RepairsComponents

A modular library of physics-based repair components for reinforcement learning environments.

## Overview

RepairsComponents provides a collection of reusable, physics-based components for building realistic repair and maintenance simulations. The library is designed to be used with reinforcement learning environments, particularly those built on the MuJoCo physics engine.

## Features

- **Fasteners**: Screws with realistic threading and fastening mechanics
- **Sockets**: Connectable components with various release mechanisms
- **Controls**: Interactive elements like buttons and switches
- **Extensible**: Easy to add new components
- **Physics-based**: Realistic interactions using MuJoCo

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/RepairsComponents-v0.git
cd RepairsComponents-v0

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
import mujoco
from repairs_components import Screw, BasicSocket, Button

# Create components
screw = Screw(thread_pitch=0.5, length=10.0)
button = Button(press_force=2.0)
socket = BasicSocket(size=8.0)

# Use in your simulation
# ...
```

## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

user_guide/installation
user_guide/quickstart
user_guide/components
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api_reference/base
api_reference/fasteners
api_reference/sockets
api_reference/controls
```

## Examples

Check out the [examples](examples/index.md) for complete usage examples.

## Contributing

Contributions are welcome! Please see our [Contributing Guide](contributing.md) for details.

## License

MIT
