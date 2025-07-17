import torch
from repairs_components.geometry.connectors.connectors import check_connections


# ---------------------------
# === check_connections() ===
# ---------------------------
def test_single_valid_connection():
    "Only one connection is close enough"
    male = {"m1": torch.tensor([[0.0, 0.0, 0.0]])}  # [B=1, 3]
    female = {"f1": torch.tensor([[1.0, 1.0, 1.0]])}  # distance ≈ 1.73 < 2.5
    result = check_connections(male, female, connection_threshold=2.5)
    assert result == [[("m1", "f1")]]


def test_single_invalid_connection():
    ""
    male = {"m1": torch.tensor([[0.0, 0.0, 0.0]])}
    female = {"f1": torch.tensor([[3.0, 0.0, 0.0]])}  # distance = 3.0 > 2.5
    result = check_connections(male, female, connection_threshold=2.5)
    assert result == [[]]


def test_multiple_connections():
    male = {
        "m1": torch.tensor([[0.0, 0.0, 0.0]]),
        "m2": torch.tensor([[5.0, 0.0, 0.0]]),
    }
    female = {
        "f1": torch.tensor([[0.5, 0.0, 0.0]]),  # close to m1
        "f2": torch.tensor([[10.0, 0.0, 0.0]]),  # too far
    }
    result = check_connections(male, female, connection_threshold=2.0)
    assert result == [[("m1", "f1")]]


def test_batch_connections():
    male = {
        "m1": torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),  # shape [2, 3]
    }
    female = {
        "f1": torch.tensor([[1.0, 0.0, 0.0], [50.0, 0.0, 0.0]]),  # shape [2, 3]
    }
    result = check_connections(male, female, connection_threshold=2.0)
    assert result == [[("m1", "f1")], []]


def test_empty_connectors():
    male = {}
    female = {}
    try:
        check_connections(male, female)
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError"


def test_only_male_or_female():
    male = {"m1": torch.tensor([[0.0, 0.0, 0.0]])}
    female = {}
    try:
        check_connections(male, female)
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError"

    male = {}
    female = {"f1": torch.tensor([[0.0, 0.0, 0.0]])}
    try:
        check_connections(male, female)
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError"


def test_multiple_matches_per_batch():
    male = {
        "m1": torch.tensor([[0.0, 0.0, 0.0]]),
        "m2": torch.tensor([[1.0, 0.0, 0.0]]),
    }
    female = {
        "f1": torch.tensor([[0.5, 0.0, 0.0]]),  # ~0.5 from both
        "f2": torch.tensor([[2.0, 0.0, 0.0]]),  # ~1.0 from m2
    }
    result = check_connections(male, female, connection_threshold=1.5)
    assert sorted(result[0]) == sorted([("m1", "f1"), ("m2", "f1"), ("m2", "f2")])
