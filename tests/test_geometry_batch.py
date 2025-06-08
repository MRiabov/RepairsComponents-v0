import torch
import pytest
from repairs_components.geometry.connectors.connectors import check_connections
from repairs_components.geometry.fasteners import check_fastener_possible_insertion


def test_check_single_fastener_possible_insertion():
    # Batch of 2: first tip matches hole0, second tip too far
    tip = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    holes = {
        'h1': torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    }
    idx = check_fastener_possible_insertion(tip, holes)
    assert torch.equal(idx, torch.tensor([0, -1]))


def test_check_fastener_possible_insertion_multiple_matches():
    # Batch of 1 with two holes: both within threshold, picks first
    tip = torch.tensor([[0.0, 0.0, 0.0]])
    holes = {
        'h1': torch.tensor([[0.6, 0.0, 0.0]]),
        'h2': torch.tensor([[0.4, 0.0, 0.0]])
    }
    idx = check_fastener_possible_insertion(tip, holes)
    assert torch.equal(idx, torch.tensor([0]))


def test_check_connections_basic():
    # Batch of 2: only env0 connectors are close
    dist_thresh = 2.5
    male = {
        'a': torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]),
        'b': torch.tensor([[1.0, 0.0, 0.0], [10.0, 9.0, 9.0]])
    }
    female = {
        'x': torch.tensor([[0.0, 1.0, 0.0], [20.0, 20.0, 20.0]])
    }
    conns = check_connections(male, female, connection_threshold=dist_thresh)
    expected = torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.long)
    # sort rows for stable order
    sorted_conns = conns[conns[:,1].argsort()]
    assert torch.equal(sorted_conns, expected)


def test_check_connections_no_match():
    # No pairs within threshold
    male = {'a': torch.tensor([[10.0, 10.0, 10.0]])}
    female = {'x': torch.tensor([[0.0, 0.0, 0.0]])}
    conns = check_connections(male, female, connection_threshold=1.0)
    assert conns.numel() == 0
