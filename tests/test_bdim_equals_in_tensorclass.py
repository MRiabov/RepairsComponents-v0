import torch
import pytest
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

    for field_name in batch_instance.to_tensordict().keys():
        batch_shape = getattr(batch_instance, field_name).shape
        stacked_shape = getattr(stacked_instance, field_name).shape
        # They should have the same shape because both have batch dim of 1
        assert batch_shape == stacked_shape, (
            f"Field {field_name}: batch_instance shape {batch_shape} != "
            f"stacked_instance shape {stacked_shape}"
        )
