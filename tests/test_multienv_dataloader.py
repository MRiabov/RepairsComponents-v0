import concurrent.futures
import pytest
import torch

from repairs_components.save_and_load.multienv_dataloader import (
    MultiEnvDataLoader,
    merge_concurrent_scene_configs as real_merge,
)


# Monkeypatch merge_concurrent_scene_configs to identity for predictable outputs
@pytest.fixture(autouse=True)
def patch_merge(monkeypatch):
    import repairs_components.save_and_load.multienv_dataloader as mdl

    monkeypatch.setattr(mdl, "merge_concurrent_scene_configs", lambda x: x)


def dummy_preproc(requests: torch.Tensor):
    # returns list of lists of strings like 'env-idx:item-index'
    return [
        [f"{idx}-{i}" for i in range(int(requests[idx]))]
        for idx in range(len(requests))
    ]


def test_get_processed_data_empty_queue():
    num_envs = 3
    dl = MultiEnvDataLoader(num_envs, dummy_preproc)
    req = torch.tensor([2, 0, 1], dtype=torch.int16)
    result = dl.get_processed_data(req)
    assert result == [["0-0", "0-1"], [], ["2-0"]]


def test_get_processed_data_invalid_length():
    dl = MultiEnvDataLoader(2, dummy_preproc)
    with pytest.raises(AssertionError):
        dl.get_processed_data(torch.tensor([1], dtype=torch.int16))


def test_get_processed_data_invalid_dtype():
    dl = MultiEnvDataLoader(2, dummy_preproc)
    with pytest.raises(AssertionError):
        dl.get_processed_data(torch.tensor([1, 2], dtype=torch.int32))


def test_populate_async_returns_future():
    dl = MultiEnvDataLoader(2, dummy_preproc)
    req = torch.tensor([1, 1], dtype=torch.int16)
    future = dl.populate_async(req)
    assert isinstance(future, concurrent.futures.Future)


def test_populate_async_invalid_args():
    dl = MultiEnvDataLoader(2, dummy_preproc)
    # wrong length
    with pytest.raises(AssertionError):
        dl.populate_async(torch.tensor([1], dtype=torch.int16))
    # wrong dtype
    with pytest.raises(AssertionError):
        dl.populate_async(torch.tensor([1, 2], dtype=torch.int32))


def test_populate_async_fills_queue():
    num_envs = 2
    dl = MultiEnvDataLoader(num_envs, dummy_preproc)
    req = torch.tensor([1, 2], dtype=torch.int16)
    future = dl.populate_async(req)
    # wait for the task to complete
    result = future.result(timeout=1.0)
    # After population, queue sizes should match req
    assert dl.prefetch_queues[0].qsize() == 1
    assert dl.prefetch_queues[1].qsize() == 2
    # Check dequeued items
    items0 = [dl.prefetch_queues[0].get() for _ in range(1)]
    items1 = [dl.prefetch_queues[1].get() for _ in range(2)]
    assert items0 == ["0-0"]
    assert items1 == ["1-0", "1-1"]
