"""Examples for using the repairs_components package."""

# This file makes the examples directory a Python package, allowing examples to import each other.

import sys
from pathlib import Path

# Add the parent directory (project root) to sys.path so src.geometry can be imported
sys.path.append(str(Path(__file__).parent.parent))
