# Installation

## Prerequisites

- Python 3.8 or higher
- MuJoCo 2.3.0 or higher
- pip (Python package manager)

## Installation Methods

### From Source (Recommended for Development)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RepairsComponents-v0.git
   cd RepairsComponents-v0
   ```

2. Install in development mode with all dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
   This installs the package in development mode, so any changes you make to the source code will be immediately available.

### Minimal Installation

If you only need the runtime dependencies:

```bash
pip install -e .
```

## Verifying Installation

You can verify that the package is installed correctly by running:

```python
import repairs_components
print(f"RepairsComponents version: {repairs_components.__version__}")
```

## Development Dependencies

For development, you'll need additional dependencies. Install them with:

```bash
pip install -r requirements-dev.txt
```

## Building Documentation

To build the documentation locally:

```bash
# Install documentation dependencies
pip install -r docs/requirements-docs.txt

# Build the docs
cd docs
make html

# Open the documentation in your browser
open _build/html/index.html  # On macOS
# or
xdg-open _build/html/index.html  # On Linux
```

## Troubleshooting

### Common Issues

1. **MuJoCo not found**:
   - Ensure MuJoCo is installed and the `MUJOCO_PATH` environment variable is set
   - On Linux, you might need to set `LD_LIBRARY_PATH`

2. **Permission errors**:
   - Use `pip install --user` or a virtual environment
   - On Linux, you might need `sudo` (not recommended)

3. **Build errors**:
   - Ensure you have a C/C++ compiler installed
   - On Windows, you might need to install Visual Studio Build Tools
