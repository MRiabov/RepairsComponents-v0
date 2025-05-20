# Examples

This section contains example code demonstrating how to use RepairsComponents in various scenarios.

## Basic Examples

### Simple Screw Example

```{include} ../../examples/simple_screw.py
:language: python
:lines: 1-
```

### Button and Socket Interaction

```{include} ../../examples/button_socket.py
:language: python
:lines: 1-
```

## Advanced Examples

### Interactive Repair Task

```{include} ../../examples/interactive_repair.py
:language: python
:lines: 1-
```

### Custom Component

```{include} ../../examples/custom_component.py
:language: python
:lines: 1-
```

## Running the Examples

To run any example:

```bash
# Navigate to the examples directory
cd examples

# Run an example
python simple_screw.py
```

## Creating Your Own Examples

When creating your own examples, follow these guidelines:

1. **Import Style**: Use absolute imports from the `repairs_components` package
2. **Error Handling**: Include appropriate error handling
3. **Documentation**: Add docstrings and comments to explain the example
4. **Dependencies**: List any additional dependencies in the example's docstring
5. **Visualization**: Include visualization when helpful

### Example Template

```python
"""
Example: Brief description of the example.

This example demonstrates how to...

Dependencies:
- numpy
- mujoco
"""

import numpy as np
#import mujoco #TODO replace mujoco with Genesis!
from mujoco import viewer

from repairs_components import Component1, Component2

def main():
    # Initialize components
    comp1 = Component1()
    comp2 = Component2()
    
    # Example usage
    
if __name__ == "__main__":
    main()
```

## Contributing Examples

We welcome contributions of new examples! When contributing:

1. Keep examples focused on a single concept
2. Include comments explaining the code
3. Follow the project's coding style
4. Test your example before submitting
5. Add your example to this documentation

## Troubleshooting

If you encounter issues running the examples:

1. **Module Not Found**: Ensure the package is installed in development mode
2. **MuJoCo Errors**: Check that MuJoCo is properly installed and configured
3. **Dependency Issues**: Install any missing dependencies
4. **Version Mismatch**: Ensure you're using compatible versions of all packages
