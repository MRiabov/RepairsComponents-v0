#!/usr/bin/env python3

import torch
from repairs_components.logic.physical_state import PhysicalState

# Create two simple PhysicalState objects
state1 = PhysicalState()
state1.body_indices = {"part1": 0, "part2": 1}
state1.position = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

state2 = PhysicalState()
state2.body_indices = {"part1": 0, "part2": 1}  # Same structure
state2.position = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

print("Before stacking:")
print(f"state1.body_indices: {state1.body_indices} (type: {type(state1.body_indices)})")
print(f"state2.body_indices: {state2.body_indices} (type: {type(state2.body_indices)})")

try:
    # Temporarily disable __post_init__ to see what happens during stacking
    original_post_init = PhysicalState.__post_init__
    PhysicalState.__post_init__ = lambda self: None

    stacked = torch.stack([state1, state2])
    print("\nAfter stacking (without post_init):")
    print(
        f"stacked.body_indices: {stacked.body_indices} (type: {type(stacked.body_indices)})"
    )
    if hasattr(stacked, "body_indices") and isinstance(stacked.body_indices, list):
        print(f"List contents: {stacked.body_indices}")

    # Restore __post_init__
    PhysicalState.__post_init__ = original_post_init

    # Now try with post_init enabled
    print("\nTrying with post_init enabled:")
    try:
        stacked_with_init = torch.stack([state1, state2])
    except Exception as e2:
        print(f"Error with post_init: {e2}")

except Exception as e:
    print(f"\nError during stacking: {e}")
    print(f"Error type: {type(e)}")
