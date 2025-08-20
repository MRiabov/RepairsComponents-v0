"""A test to verify stacking behavior in tensorclasses."""

import torch
from tensordict import TensorClass


# Dummy example class
class Class123(TensorClass):
    x: torch.Tensor = torch.tensor(0)
    y: torch.Tensor = torch.tensor([1, 2, 3])


def test_batch_size_vs_stack_shapes():
    batch_instance = Class123(batch_size=None)
    stacked_instance = torch.stack(
        [Class123(batch_size=None)]
    )  # returns Class123 with batch dim

    # Both should be instances of Class123
    assert isinstance(batch_instance, Class123)
    assert isinstance(stacked_instance, Class123)

    assert len(stacked_instance.x.shape) == len(batch_instance.x.shape) + 1
