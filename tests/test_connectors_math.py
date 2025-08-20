import torch

from repairs_components.geometry.connectors.connectors import check_connections


# ---------------------------
# === check_connections() ===
# ---------------------------
def test_single_valid_connection():
    "Only one connection is close enough"
    male = torch.tensor([[0.0, 0.0, 0.0]])  # [1, 3]
    female = torch.tensor([[1.0, 1.0, 1.0]])  # distance ≈ 1.73 < 2.5
    result = check_connections(male, female, connection_threshold=2.5)
    expected = torch.tensor([[0, 0]])  # male_idx=0, female_idx=0
    assert torch.equal(result, expected)


def test_single_invalid_connection():
    "No connection due to distance"
    male = torch.tensor([[0.0, 0.0, 0.0]])
    female = torch.tensor([[3.0, 0.0, 0.0]])  # distance = 3.0 > 2.5
    result = check_connections(male, female, connection_threshold=2.5)
    expected = torch.empty((0, 2), dtype=torch.long)  # no connections
    assert torch.equal(result, expected)


def test_multiple_connections():
    male = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # m_idx=0
            [5.0, 0.0, 0.0],  # m_idx=1
        ]
    )
    female = torch.tensor(
        [
            [0.5, 0.0, 0.0],  # f_idx=0, close to m_idx=0
            [10.0, 0.0, 0.0],  # f_idx=1, too far from both
        ]
    )
    result = check_connections(male, female, connection_threshold=2.0)
    expected = torch.tensor([[0, 0]])  # only m_idx=0 connects to f_idx=0
    assert torch.equal(result, expected)


def test_batch_connections():
    # Test multiple male and female connectors
    male = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # m_idx=0
            [10.0, 0.0, 0.0],  # m_idx=1
        ]
    )
    female = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # f_idx=0, close to m_idx=0
            [50.0, 0.0, 0.0],  # f_idx=1, too far from both
        ]
    )
    result = check_connections(male, female, connection_threshold=2.0)
    expected = torch.tensor([[0, 0]])  # only m_idx=0 connects to f_idx=0
    assert torch.equal(result, expected)


def test_empty_connectors():
    # Test with empty tensors
    male = torch.empty((0, 3))
    female = torch.empty((0, 3))
    result = check_connections(male, female)
    expected = torch.empty((0, 2), dtype=torch.long)
    assert torch.equal(result, expected)


def test_only_male_or_female():
    # Test with only male connectors
    male = torch.tensor([[0.0, 0.0, 0.0]])
    female = torch.empty((0, 3))
    result = check_connections(male, female)
    expected = torch.empty((0, 2), dtype=torch.long)
    assert torch.equal(result, expected)

    # Test with only female connectors
    male = torch.empty((0, 3))
    female = torch.tensor([[0.0, 0.0, 0.0]])
    result = check_connections(male, female)
    expected = torch.empty((0, 2), dtype=torch.long)
    assert torch.equal(result, expected)


def test_multiple_matches_per_batch():
    male = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # m_idx=0
            [1.0, 0.0, 0.0],  # m_idx=1
        ]
    )
    female = torch.tensor(
        [
            [0.5, 0.0, 0.0],  # f_idx=0, ~0.5 from both males
            [2.0, 0.0, 0.0],  # f_idx=1, ~1.0 from m_idx=1
        ]
    )
    result = check_connections(male, female, connection_threshold=1.5)
    # Expected connections: (m0,f0), (m1,f0), (m1,f1)
    expected_connections = {(0, 0), (1, 0), (1, 1)}
    result_connections = {(int(row[0]), int(row[1])) for row in result}
    assert result_connections == expected_connections
