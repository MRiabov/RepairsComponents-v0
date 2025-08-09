import pytest
import torch

from repairs_components.logic.physical_state import (
    PhysicalState,
    register_fasteners_batch,
    register_bodies_batch,
    update_bodies_batch,
)
from repairs_components.geometry.fasteners import get_fastener_singleton_name


@pytest.fixture
def single_physical_state():
    """Create a single PhysicalState with some bodies for testing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = PhysicalState(device=device)

    # Add some bodies using register_bodies_batch approach
    # We'll create the body indices manually and set tensors
    num_bodies = 3
    num_holes = 6

    # Create body indices
    for i in range(num_bodies):
        body_name = f"body_{i}@solid"
        state.body_indices[body_name] = i
        state.inverse_body_indices[i] = body_name

    # Set tensors for single state (no batch dimension yet)
    state.position = torch.rand(num_bodies, 3, device=device) * 0.32 - 0.16
    state.position[:, 2] = torch.abs(state.position[:, 2])  # Ensure z >= 0

    quat = torch.randn(num_bodies, 4, device=device)
    state.quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    state.fixed = torch.zeros(num_bodies, dtype=torch.bool, device=device)
    state.count_fasteners_held = torch.zeros(
        num_bodies, dtype=torch.int8, device=device
    )

    # Set part_hole_batch mapping holes to bodies
    state.part_hole_batch = torch.randint(0, num_bodies, (num_holes,), device=device)

    return state


class TestRegisterFastenersBatch:
    """Test suite for register_fasteners_batch function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_batched_physical_state(self, batch_size: int = 2, single_state=None):
        """Create a batched PhysicalState by stacking single states."""
        if single_state is None:
            # Create a simple single state
            single_state = PhysicalState(device=self.device)

            # Add some bodies
            num_bodies = 3
            num_holes = 6

            for i in range(num_bodies):
                body_name = f"body_{i}@solid"
                single_state.body_indices[body_name] = i
                single_state.inverse_body_indices[i] = body_name

            # Set minimal tensors
            single_state.position = (
                torch.rand(num_bodies, 3, device=self.device) * 0.32 - 0.16
            )
            single_state.position[:, 2] = torch.abs(single_state.position[:, 2])

            quat = torch.randn(num_bodies, 4, device=self.device)
            single_state.quat = quat / torch.norm(quat, dim=-1, keepdim=True)

            single_state.fixed = torch.zeros(
                num_bodies, dtype=torch.bool, device=self.device
            )
            single_state.count_fasteners_held = torch.zeros(
                num_bodies, dtype=torch.int8, device=self.device
            )
            single_state.part_hole_batch = torch.randint(
                0, num_bodies, (num_holes,), device=self.device
            )

        # Stack to create batched state
        states = [single_state for _ in range(batch_size)]
        batched_state: PhysicalState = torch.stack(states)  # type: ignore

        return batched_state

    def test_register_fasteners_batch_basic(self):
        """Test basic functionality of register_fasteners_batch."""
        batch_size = 2
        num_fasteners = 3
        physical_state = self.create_batched_physical_state(batch_size=batch_size)

        # Create test data
        fastener_pos = torch.randn(batch_size, num_fasteners, 3)
        fastener_quat = torch.randn(batch_size, num_fasteners, 4)
        fastener_init_hole_a = torch.randint(0, 6, (batch_size, num_fasteners))
        fastener_init_hole_b = torch.randint(0, 6, (batch_size, num_fasteners))

        # Ensure holes are different (except for -1 case)
        fastener_init_hole_b = torch.where(
            fastener_init_hole_a == fastener_init_hole_b,
            (fastener_init_hole_a + 1) % 6,
            fastener_init_hole_b,
        )

        fastener_compound_names = [
            get_fastener_singleton_name(3.0, 10.0),
            get_fastener_singleton_name(4.0, 12.0),
            get_fastener_singleton_name(5.0, 16.0),
        ]

        # Call the function
        result_state = register_fasteners_batch(
            physical_state,
            fastener_pos,
            fastener_quat,
            fastener_init_hole_a,
            fastener_init_hole_b,
            fastener_compound_names,
        )

        # Verify the result
        assert result_state is physical_state  # Should modify in place
        assert result_state.fasteners_pos.shape == (batch_size, num_fasteners, 3)
        assert result_state.fasteners_quat.shape == (batch_size, num_fasteners, 4)
        assert result_state.fasteners_diam.shape == (batch_size, num_fasteners)
        assert result_state.fasteners_length.shape == (batch_size, num_fasteners)
        assert result_state.fasteners_attached_to_hole.shape == (
            batch_size,
            num_fasteners,
            2,
        )
        assert result_state.fasteners_attached_to_body.shape == (
            batch_size,
            num_fasteners,
            2,
        )
        assert result_state.count_fasteners_held.shape == (batch_size, 3)  # num_bodies

    def test_register_fasteners_batch_fastener_params(self):
        """Test that fastener parameters are correctly extracted from names."""
        batch_size = 1
        num_fasteners = 2
        physical_state = self.create_batched_physical_state(batch_size=batch_size)

        fastener_pos = torch.randn(batch_size, num_fasteners, 3)
        fastener_quat = torch.randn(batch_size, num_fasteners, 4)
        fastener_init_hole_a = torch.tensor([[0, 1]])
        fastener_init_hole_b = torch.tensor([[2, 3]])
        fastener_compound_names = [
            get_fastener_singleton_name(3.0, 10.0),
            get_fastener_singleton_name(4.0, 12.0),
        ]

        result_state = register_fasteners_batch(
            physical_state,
            fastener_pos,
            fastener_quat,
            fastener_init_hole_a,
            fastener_init_hole_b,
            fastener_compound_names,
        )

        # Verify fastener parameters are set correctly
        # get_fastener_singleton_name uses millimeters; PhysicalState stores values in meters
        expected_diam_0, expected_length_0 = 3.0 / 1000.0, 10.0 / 1000.0
        expected_diam_1, expected_length_1 = 4.0 / 1000.0, 12.0 / 1000.0

        assert torch.allclose(
            result_state.fasteners_diam[0, 0], torch.tensor(expected_diam_0)
        )
        assert torch.allclose(
            result_state.fasteners_length[0, 0], torch.tensor(expected_length_0)
        )
        assert torch.allclose(
            result_state.fasteners_diam[0, 1], torch.tensor(expected_diam_1)
        )
        assert torch.allclose(
            result_state.fasteners_length[0, 1], torch.tensor(expected_length_1)
        )

    def test_register_fasteners_batch_hole_attachment(self):
        """Test that fasteners are correctly attached to holes and bodies."""
        batch_size = 1
        num_fasteners = 2
        physical_state = self.create_batched_physical_state(batch_size=batch_size)

        # Set specific hole-to-body mapping for predictable testing
        physical_state.part_hole_batch = torch.tensor(
            [[0, 1, 2, 0, 1, 2]]
        )  # holes map to bodies

        fastener_pos = torch.randn(batch_size, num_fasteners, 3)
        fastener_quat = torch.randn(batch_size, num_fasteners, 4)
        fastener_init_hole_a = torch.tensor([[0, 2]])  # holes 0, 2
        fastener_init_hole_b = torch.tensor([[1, 3]])  # holes 1, 3
        fastener_compound_names = [
            get_fastener_singleton_name(3.0, 10.0),
            get_fastener_singleton_name(4.0, 12.0),
        ]

        result_state = register_fasteners_batch(
            physical_state,
            fastener_pos,
            fastener_quat,
            fastener_init_hole_a,
            fastener_init_hole_b,
            fastener_compound_names,
        )

        # Verify hole attachments
        expected_holes = torch.tensor([[[0, 1], [2, 3]]])
        assert torch.equal(result_state.fasteners_attached_to_hole, expected_holes)

        # Verify body attachments (based on part_hole_batch mapping)
        expected_bodies = torch.tensor(
            [[[0, 1], [2, 0]]]
        )  # holes 0,1,2,3 -> bodies 0,1,2,0
        assert torch.equal(result_state.fasteners_attached_to_body, expected_bodies)

    def test_register_fasteners_batch_empty_holes(self):
        """Test handling of empty holes (-1 values)."""
        batch_size = 1
        num_fasteners = 2
        physical_state = self.create_batched_physical_state(batch_size=batch_size)

        fastener_pos = torch.randn(batch_size, num_fasteners, 3)
        fastener_quat = torch.randn(batch_size, num_fasteners, 4)
        fastener_init_hole_a = torch.tensor([[-1, 0]])  # First fastener unattached
        fastener_init_hole_b = torch.tensor([[-1, 1]])  # First fastener unattached
        fastener_compound_names = [
            get_fastener_singleton_name(3.0, 10.0),
            get_fastener_singleton_name(4.0, 12.0),
        ]

        result_state = register_fasteners_batch(
            physical_state,
            fastener_pos,
            fastener_quat,
            fastener_init_hole_a,
            fastener_init_hole_b,
            fastener_compound_names,
        )

        # Verify empty holes are handled correctly
        assert result_state.fasteners_attached_to_hole[0, 0, 0] == -1
        assert result_state.fasteners_attached_to_hole[0, 0, 1] == -1
        assert result_state.fasteners_attached_to_body[0, 0, 0] == -1
        assert result_state.fasteners_attached_to_body[0, 0, 1] == -1

    def test_register_fasteners_batch_fastener_count(self):
        """Test that fastener counts per body are correctly calculated."""
        batch_size = 1
        num_fasteners = 3
        physical_state = self.create_batched_physical_state(batch_size=batch_size)

        # Set specific hole-to-body mapping
        physical_state.part_hole_batch = torch.tensor(
            [[0, 0, 1, 1, 2, 2]]
        )  # 2 holes per body

        fastener_pos = torch.randn(batch_size, num_fasteners, 3)
        fastener_quat = torch.randn(batch_size, num_fasteners, 4)
        # Fastener 0: body 0 to body 1, Fastener 1: body 1 to body 2, Fastener 2: body 0 to body 2
        fastener_init_hole_a = torch.tensor(
            [[0, 2, 1]]
        )  # holes 0, 2, 1 -> bodies 0, 1, 0
        fastener_init_hole_b = torch.tensor(
            [[3, 4, 5]]
        )  # holes 3, 4, 5 -> bodies 1, 2, 2
        fastener_compound_names = [
            get_fastener_singleton_name(3.0, 10.0),
            get_fastener_singleton_name(4.0, 12.0),
            get_fastener_singleton_name(5.0, 16.0),
        ]

        result_state = register_fasteners_batch(
            physical_state,
            fastener_pos,
            fastener_quat,
            fastener_init_hole_a,
            fastener_init_hole_b,
            fastener_compound_names,
        )

        # Body 0: connected to fasteners 0, 2 -> count = 2
        # Body 1: connected to fasteners 0, 1 -> count = 2
        # Body 2: connected to fasteners 1, 2 -> count = 2
        expected_counts = torch.tensor([[2, 2, 2]], dtype=torch.int8)
        assert torch.equal(result_state.count_fasteners_held, expected_counts)

    def test_register_fasteners_batch_shape_validation(self):
        """Test input shape validation."""
        batch_size = 2
        num_fasteners = 3
        physical_state = self.create_batched_physical_state(batch_size=batch_size)

        # Test wrong fastener_pos shape
        with pytest.raises(AssertionError, match="Expected fastener_pos shape"):
            register_fasteners_batch(
                physical_state,
                torch.randn(batch_size, num_fasteners + 1, 3),  # Wrong num_fasteners
                torch.randn(batch_size, num_fasteners, 4),
                torch.randint(0, 6, (batch_size, num_fasteners)),
                torch.randint(0, 6, (batch_size, num_fasteners)),
                [get_fastener_singleton_name(3.0, 10.0)] * num_fasteners,
            )

        # Test wrong fastener_quat shape
        with pytest.raises(AssertionError, match="Expected fastener_quat shape"):
            register_fasteners_batch(
                physical_state,
                torch.randn(batch_size, num_fasteners, 3),
                torch.randn(batch_size, num_fasteners, 3),  # Wrong last dimension
                torch.randint(0, 6, (batch_size, num_fasteners)),
                torch.randint(0, 6, (batch_size, num_fasteners)),
                [get_fastener_singleton_name(3.0, 10.0)] * num_fasteners,
            )

    def test_register_fasteners_batch_hole_validation(self):
        """Test hole index validation."""
        batch_size = 1
        num_fasteners = 2
        # Create state with only 4 holes instead of default 6
        single_state = PhysicalState(device=self.device)
        num_bodies = 3
        num_holes = 4  # Reduced number of holes

        for i in range(num_bodies):
            body_name = f"body_{i}@solid"
            single_state.body_indices[body_name] = i
            single_state.inverse_body_indices[i] = body_name

        single_state.position = (
            torch.rand(num_bodies, 3, device=self.device) * 0.32 - 0.16
        )
        single_state.position[:, 2] = torch.abs(single_state.position[:, 2])

        quat = torch.randn(num_bodies, 4, device=self.device)
        single_state.quat = quat / torch.norm(quat, dim=-1, keepdim=True)

        single_state.fixed = torch.zeros(
            num_bodies, dtype=torch.bool, device=self.device
        )
        single_state.count_fasteners_held = torch.zeros(
            num_bodies, dtype=torch.int8, device=self.device
        )
        single_state.part_hole_batch = torch.randint(
            0, num_bodies, (num_holes,), device=self.device
        )

        physical_state: PhysicalState = torch.stack([single_state])  # type: ignore

        fastener_pos = torch.randn(batch_size, num_fasteners, 3)
        fastener_quat = torch.randn(batch_size, num_fasteners, 4)
        fastener_compound_names = [
            get_fastener_singleton_name(3.0, 10.0),
            get_fastener_singleton_name(4.0, 12.0),
        ]

        # Test out-of-range hole indices (now hole 4 is out of range since we only have 4 holes: 0,1,2,3)
        with pytest.raises(
            AssertionError, match="fastener_init_hole_a are out of range"
        ):
            register_fasteners_batch(
                physical_state,
                fastener_pos,
                fastener_quat,
                torch.tensor([[0, 4]]),  # hole 4 is out of range (max is 3)
                torch.tensor([[1, 2]]),
                fastener_compound_names,
            )

        with pytest.raises(
            AssertionError, match="fastener_init_hole_b are out of range"
        ):
            register_fasteners_batch(
                physical_state,
                fastener_pos,
                fastener_quat,
                torch.tensor([[0, 1]]),
                torch.tensor([[2, 5]]),  # hole 5 is out of range (max is 3)
                fastener_compound_names,
            )

    def test_register_fasteners_batch_same_hole_validation(self):
        """Test validation that fastener holes must be different (unless both are -1)."""
        batch_size = 1
        num_fasteners = 2
        physical_state = self.create_batched_physical_state(batch_size=batch_size)

        fastener_pos = torch.randn(batch_size, num_fasteners, 3)
        fastener_quat = torch.randn(batch_size, num_fasteners, 4)
        fastener_compound_names = [
            get_fastener_singleton_name(3.0, 10.0),
            get_fastener_singleton_name(4.0, 12.0),
        ]

        # Test same hole indices (should fail)
        with pytest.raises(
            AssertionError, match="Fastener init holes must be different or empty"
        ):
            register_fasteners_batch(
                physical_state,
                fastener_pos,
                fastener_quat,
                torch.tensor([[0, 1]]),
                torch.tensor([[0, 2]]),  # First fastener has same hole for both ends
                fastener_compound_names,
            )

        # Test both holes as -1 (should pass)
        result_state = register_fasteners_batch(
            physical_state,
            fastener_pos,
            fastener_quat,
            torch.tensor([[-1, 0]]),
            torch.tensor([[-1, 1]]),  # First fastener unattached (both -1)
            fastener_compound_names,
        )
        assert result_state is not None

    def test_register_fasteners_batch_no_part_hole_batch(self):
        """Test that function fails when part_hole_batch is not set."""
        # Create a batched state but with part_hole_batch set to None
        single_state = PhysicalState(device=self.device)
        single_state.body_indices = {"body_0@solid": 0}
        single_state.inverse_body_indices = {0: "body_0@solid"}

        # Set minimal required tensors
        single_state.position = torch.rand(1, 3, device=self.device) * 0.32 - 0.16
        single_state.position[:, 2] = torch.abs(single_state.position[:, 2])

        quat = torch.randn(1, 4, device=self.device)
        single_state.quat = quat / torch.norm(quat, dim=-1, keepdim=True)

        single_state.fixed = torch.zeros(1, dtype=torch.bool, device=self.device)
        single_state.count_fasteners_held = torch.zeros(
            1, dtype=torch.int8, device=self.device
        )

        # Don't set part_hole_batch (it will be None by default)

        physical_state: PhysicalState = torch.stack([single_state])  # type: ignore

        # The function should fail when part_hole_batch is None, but the actual error
        # might be a RuntimeError due to the error message formatting
        with pytest.raises((AssertionError, RuntimeError)):
            register_fasteners_batch(
                physical_state,
                torch.randn(1, 1, 3),
                torch.randn(1, 1, 4),
                torch.tensor([[0]]),
                torch.tensor([[1]]),
                [get_fastener_singleton_name(3.0, 10.0)],
            )

    def test_register_fasteners_batch_fastener_as_body_name(self):
        """Test that fastener compound names cannot be registered as body names."""
        batch_size = 1
        num_fasteners = 1
        physical_state = self.create_batched_physical_state(batch_size=batch_size)

        # Add a fastener name as a body name
        fastener_name = get_fastener_singleton_name(3.0, 10.0)
        physical_state.body_indices[fastener_name] = len(physical_state.body_indices)

        fastener_pos = torch.randn(batch_size, num_fasteners, 3)
        fastener_quat = torch.randn(batch_size, num_fasteners, 4)
        fastener_init_hole_a = torch.tensor([[0]])
        fastener_init_hole_b = torch.tensor([[1]])

        with pytest.raises(
            AssertionError, match="Fasteners can't be registered as bodies"
        ):
            register_fasteners_batch(
                physical_state,
                fastener_pos,
                fastener_quat,
                fastener_init_hole_a,
                fastener_init_hole_b,
                [fastener_name],
            )

    def test_register_fasteners_batch_device_handling(self):
        """Test that tensors are moved to the correct device."""
        batch_size = 1
        num_fasteners = 1
        physical_state = self.create_batched_physical_state(batch_size=batch_size)

        # The physical_state is already on the target device (CPU)
        target_device = torch.device("cpu")

        # Create tensors on different device (if CUDA available)
        if torch.cuda.is_available():
            fastener_pos = torch.randn(batch_size, num_fasteners, 3, device="cuda")
            fastener_quat = torch.randn(batch_size, num_fasteners, 4, device="cuda")
        else:
            fastener_pos = torch.randn(batch_size, num_fasteners, 3)
            fastener_quat = torch.randn(batch_size, num_fasteners, 4)

        fastener_init_hole_a = torch.tensor([[0]])
        fastener_init_hole_b = torch.tensor([[1]])
        fastener_compound_names = [get_fastener_singleton_name(3.0, 10.0)]

        result_state = register_fasteners_batch(
            physical_state,
            fastener_pos,
            fastener_quat,
            fastener_init_hole_a,
            fastener_init_hole_b,
            fastener_compound_names,
        )

        # Verify tensors are on the correct device
        assert result_state.fasteners_pos.device == target_device
        assert result_state.fasteners_quat.device == target_device


class TestRegisterBodiesBatch:
    """Test suite for register_bodies_batch function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_empty_batched_physical_state(self, batch_size: int = 2):
        """Create an empty batched PhysicalState."""
        # Create a simple single state
        single_state = PhysicalState(device=self.device)

        # Initialize empty tensors
        single_state.position = torch.empty((0, 3), device=self.device)
        single_state.quat = torch.empty((0, 4), device=self.device)
        single_state.fixed = torch.empty((0,), dtype=torch.bool, device=self.device)
        single_state.count_fasteners_held = torch.empty(
            (0,), dtype=torch.int8, device=self.device
        )
        single_state.male_connector_positions = torch.empty((0, 3), device=self.device)
        single_state.female_connector_positions = torch.empty(
            (0, 3), device=self.device
        )
        single_state.male_connector_batch = torch.empty(
            (0,), dtype=torch.long, device=self.device
        )
        single_state.female_connector_batch = torch.empty(
            (0,), dtype=torch.long, device=self.device
        )
        single_state.connector_indices_from_name = {}

        # Stack to create batched state
        states = [single_state for _ in range(batch_size)]
        batched_state: PhysicalState = torch.stack(states)  # type: ignore

        return batched_state

    def test_register_bodies_batch_basic(self):
        """Test basic functionality of register_bodies_batch."""
        batch_size = 2
        num_bodies = 3
        physical_state = self.create_empty_batched_physical_state(batch_size=batch_size)

        # Create test data
        names = ["body_0@solid", "body_1@fixed_solid", "body_2@solid"]
        positions = (
            torch.randn(batch_size, num_bodies, 3) * 0.1
        )  # Keep well within bounds
        positions[:, :, 2] = (
            torch.abs(positions[:, :, 2]) + 0.1
        )  # Ensure z > 0 and within bounds
        rotations = torch.randn(batch_size, num_bodies, 4)
        rotations = rotations / torch.norm(rotations, dim=-1, keepdim=True)  # Normalize
        fixed = torch.tensor([False, True, False])  # [num_bodies]

        # Register bodies
        result_state = register_bodies_batch(
            physical_state,
            names,
            positions,
            rotations,
            fixed,
        )

        # Verify body indices were updated
        assert len(result_state.body_indices) == num_bodies
        for i, name in enumerate(names):
            assert name in result_state.body_indices
            assert result_state.body_indices[name] == i
            assert result_state.inverse_body_indices[i] == name

        # Verify tensors were updated
        assert result_state.position.shape == (batch_size, num_bodies, 3)
        assert result_state.quat.shape == (batch_size, num_bodies, 4)
        assert result_state.fixed.shape == (batch_size, num_bodies)
        assert result_state.count_fasteners_held.shape == (batch_size, num_bodies)

        # Verify fixed values are correct
        expected_fixed = fixed.unsqueeze(0).expand(batch_size, num_bodies)
        assert torch.equal(result_state.fixed, expected_fixed)

        # Verify positions and rotations match
        assert torch.allclose(result_state.position, positions)
        assert torch.allclose(result_state.quat, rotations)

    def test_register_bodies_batch_with_connectors(self):
        """Test register_bodies_batch with connector bodies."""
        batch_size = 2
        num_bodies = 2
        physical_state = self.create_empty_batched_physical_state(batch_size=batch_size)

        # Create test data with connectors
        names = ["connector_male@connector", "connector_female@connector"]
        positions = torch.randn(batch_size, num_bodies, 3) * 0.1
        positions[:, :, 2] = torch.abs(positions[:, :, 2]) + 0.1
        rotations = torch.randn(batch_size, num_bodies, 4)
        rotations = rotations / torch.norm(rotations, dim=-1, keepdim=True)
        fixed = torch.tensor([False, False])

        # Create connector relative positions
        connector_positions_relative = torch.tensor(
            [
                [0.1, 0.0, 0.05],  # male connector
                [-0.1, 0.0, 0.05],  # female connector
            ]
        )

        # Register bodies
        result_state = register_bodies_batch(
            physical_state,
            names,
            positions,
            rotations,
            fixed,
            connector_positions_relative,
        )

        # Verify body indices were updated
        assert len(result_state.body_indices) == num_bodies
        for i, name in enumerate(names):
            assert name in result_state.body_indices

        # Verify connector positions were calculated
        # We should have one male and one female connector
        assert result_state.male_connector_positions.shape == (batch_size, 1, 3)
        assert result_state.female_connector_positions.shape == (batch_size, 1, 3)

    def test_register_bodies_batch_mixed_connectors(self):
        """Test register_bodies_batch with mixed connector and non-connector bodies."""
        batch_size = 2
        num_bodies = 3
        physical_state = self.create_empty_batched_physical_state(batch_size=batch_size)

        # Create test data with mixed body types
        names = ["body_0@solid", "connector_male@connector", "body_2@fixed_solid"]
        positions = torch.randn(batch_size, num_bodies, 3) * 0.1
        positions[:, :, 2] = torch.abs(positions[:, :, 2]) + 0.1
        rotations = torch.randn(batch_size, num_bodies, 4)
        rotations = rotations / torch.norm(rotations, dim=-1, keepdim=True)
        fixed = torch.tensor([False, False, True])

        # Create connector relative positions (NaN for non-connectors)
        connector_positions_relative = torch.tensor(
            [
                [float("nan"), float("nan"), float("nan")],  # non-connector
                [0.1, 0.0, 0.05],  # connector
                [float("nan"), float("nan"), float("nan")],  # non-connector
            ]
        )

        # Register bodies
        result_state = register_bodies_batch(
            physical_state,
            names,
            positions,
            rotations,
            fixed,
            connector_positions_relative,
        )

        # Verify body indices were updated
        assert len(result_state.body_indices) == num_bodies

        # Verify connector positions were calculated only for connector (index 1)
        # We should have one male connector and no female connectors
        assert result_state.male_connector_positions.shape == (
            batch_size,
            1,
            3,
        )  # Only one male connector
        assert result_state.female_connector_positions.shape == (
            batch_size,
            0,
            3,
        )  # No female connectors

    def test_register_bodies_batch_input_validation(self):
        """Test input validation for register_bodies_batch."""
        batch_size = 2
        num_bodies = 2
        physical_state = self.create_empty_batched_physical_state(batch_size=batch_size)

        names = ["body_0@solid", "body_1@solid"]
        positions = torch.randn(batch_size, num_bodies, 3) * 0.1
        positions[:, :, 2] = torch.abs(positions[:, :, 2]) + 0.1
        rotations = torch.randn(batch_size, num_bodies, 4)
        rotations = rotations / torch.norm(rotations, dim=-1, keepdim=True)
        fixed = torch.tensor([False, False])

        # Test wrong positions shape
        with pytest.raises(AssertionError, match="Expected positions shape"):
            register_bodies_batch(
                physical_state,
                names,
                positions[:, :1],  # Wrong num_bodies dimension
                rotations,
                fixed,
            )

        # Test wrong rotations shape
        with pytest.raises(AssertionError, match="Expected rotations shape"):
            register_bodies_batch(
                physical_state,
                names,
                positions,
                rotations[:, :, :3],  # Wrong quaternion dimension
                fixed,
            )

        # Test wrong fixed shape
        with pytest.raises(ValueError, match="Expected `fixed` shape"):
            register_bodies_batch(
                physical_state,
                names,
                positions,
                rotations,
                torch.tensor([False]),  # Wrong number of bodies
            )

    def test_register_bodies_batch_name_validation(self):
        """Test name format validation."""
        batch_size = 1
        num_bodies = 1
        physical_state = self.create_empty_batched_physical_state(batch_size=batch_size)

        positions = torch.randn(batch_size, num_bodies, 3) * 0.1
        positions[:, :, 2] = torch.abs(positions[:, :, 2]) + 0.1
        rotations = torch.randn(batch_size, num_bodies, 4)
        rotations = rotations / torch.norm(rotations, dim=-1, keepdim=True)
        fixed = torch.tensor([False])

        # Test invalid name format
        with pytest.raises(AssertionError, match="Body name must end with"):
            register_bodies_batch(
                physical_state,
                ["invalid_name"],
                positions,
                rotations,
                fixed,
            )

    def test_register_bodies_batch_duplicate_names(self):
        """Test that duplicate body names are rejected."""
        batch_size = 1
        num_bodies = 1
        physical_state = self.create_empty_batched_physical_state(batch_size=batch_size)

        # Add a body first
        physical_state.body_indices["body_0@solid"] = 0

        positions = torch.randn(batch_size, num_bodies, 3) * 0.1
        positions[:, :, 2] = torch.abs(positions[:, :, 2]) + 0.1
        rotations = torch.randn(batch_size, num_bodies, 4)
        rotations = rotations / torch.norm(rotations, dim=-1, keepdim=True)
        fixed = torch.tensor([False])

        # Test duplicate name
        with pytest.raises(AssertionError, match="already registered"):
            register_bodies_batch(
                physical_state,
                ["body_0@solid"],  # Duplicate name
                positions,
                rotations,
                fixed,
            )

    def test_register_bodies_batch_connector_validation(self):
        """Test connector-specific validation."""
        batch_size = 1
        num_bodies = 1
        physical_state = self.create_empty_batched_physical_state(batch_size=batch_size)

        names = ["connector_male@connector"]
        positions = torch.randn(batch_size, num_bodies, 3) * 0.1
        positions[:, :, 2] = torch.abs(positions[:, :, 2]) + 0.1
        rotations = torch.randn(batch_size, num_bodies, 4)
        rotations = rotations / torch.norm(rotations, dim=-1, keepdim=True)
        fixed = torch.tensor([False])

        # Test NaN connector position for connector body
        connector_positions_relative = torch.tensor(
            [[float("nan"), float("nan"), float("nan")]]
        )

        with pytest.raises(
            AssertionError,
            match="must have valid connector_position_relative_to_center",
        ):
            register_bodies_batch(
                physical_state,
                names,
                positions,
                rotations,
                fixed,
                connector_positions_relative,
            )

    def test_register_bodies_batch_fixed_tensor_shapes(self):
        """Test different shapes for fixed tensor."""
        batch_size = 2
        num_bodies = 2
        physical_state = self.create_empty_batched_physical_state(batch_size=batch_size)

        names = ["body_0@solid", "body_1@fixed_solid"]
        positions = torch.randn(batch_size, num_bodies, 3) * 0.1
        positions[:, :, 2] = torch.abs(positions[:, :, 2]) + 0.1
        rotations = torch.randn(batch_size, num_bodies, 4)
        rotations = rotations / torch.norm(rotations, dim=-1, keepdim=True)

        # Test with [num_bodies] shape
        fixed_1d = torch.tensor([False, True])
        result_state = register_bodies_batch(
            physical_state,
            names,
            positions,
            rotations,
            fixed_1d,
        )
        expected_fixed = fixed_1d.unsqueeze(0).expand(batch_size, num_bodies)
        assert torch.equal(result_state.fixed, expected_fixed)

        # Reset state
        physical_state = self.create_empty_batched_physical_state(batch_size=batch_size)

        # Test with [B, num_bodies] shape
        fixed_2d = torch.tensor([[False, True], [True, False]])
        result_state = register_bodies_batch(
            physical_state,
            names,
            positions,
            rotations,
            fixed_2d,
        )
        assert torch.equal(result_state.fixed, fixed_2d)

    def test_register_bodies_batch_device_handling(self):
        """Test that tensors are moved to the correct device."""
        batch_size = 1
        num_bodies = 1
        physical_state = self.create_empty_batched_physical_state(batch_size=batch_size)

        target_device = torch.device("cpu")

        names = ["body_0@solid"]
        positions = torch.randn(batch_size, num_bodies, 3) * 0.1
        positions[:, :, 2] = torch.abs(positions[:, :, 2]) + 0.1
        rotations = torch.randn(batch_size, num_bodies, 4)
        rotations = rotations / torch.norm(rotations, dim=-1, keepdim=True)
        fixed = torch.tensor([False])

        # Create tensors on different device (if CUDA available)
        if torch.cuda.is_available():
            positions = positions.to("cuda")
            rotations = rotations.to("cuda")
            fixed = fixed.to("cuda")

        result_state = register_bodies_batch(
            physical_state,
            names,
            positions,
            rotations,
            fixed,
        )

        # Verify tensors are on the correct device
        assert result_state.position.device == target_device
        assert result_state.quat.device == target_device
        assert result_state.fixed.device == target_device


class TestUpdateBodiesBatch:
    """Tests for update_bodies_batch function."""

    def setup_method(self):
        self.device = torch.device("cpu")

    def create_registered_state(
        self, batch_size: int = 2, include_connectors: bool = False
    ):
        """Create a batched state with registered bodies (optionally connectors)."""
        # Empty batched state
        single_state = PhysicalState(device=self.device)
        single_state.position = torch.empty((0, 3), device=self.device)
        single_state.quat = torch.empty((0, 4), device=self.device)
        single_state.fixed = torch.empty((0,), dtype=torch.bool, device=self.device)
        single_state.count_fasteners_held = torch.empty(
            (0,), dtype=torch.int8, device=self.device
        )
        single_state.male_connector_positions = torch.empty((0, 3), device=self.device)
        single_state.female_connector_positions = torch.empty(
            (0, 3), device=self.device
        )
        single_state.male_connector_batch = torch.empty(
            (0,), dtype=torch.long, device=self.device
        )
        single_state.female_connector_batch = torch.empty(
            (0,), dtype=torch.long, device=self.device
        )
        single_state.connector_indices_from_name = {}

        states = [single_state for _ in range(batch_size)]
        batched_state: PhysicalState = torch.stack(states)  # type: ignore

        if include_connectors:
            names = [
                "body_0@solid",
                "connector_male@connector",
                "connector_female@connector",
            ]
            num_bodies = len(names)
        else:
            names = ["body_0@solid", "body_1@fixed_solid", "body_2@solid"]
            num_bodies = len(names)

        # Initial positions/rotations
        positions = torch.zeros(batch_size, num_bodies, 3)
        # Identity quaternions for simplicity
        rotations = torch.zeros(batch_size, num_bodies, 4)
        rotations[..., 0] = 1.0

        if include_connectors:
            fixed = torch.tensor([False, False, False])
            connector_rel = torch.tensor(
                [
                    [float("nan"), float("nan"), float("nan")],
                    [0.10, 0.00, 0.05],
                    [-0.10, 0.00, 0.05],
                ]
            )
            register_bodies_batch(
                batched_state, names, positions, rotations, fixed, connector_rel
            )
        else:
            fixed = torch.tensor([False, True, False])
            register_bodies_batch(batched_state, names, positions, rotations, fixed)

        return batched_state, names

    def test_update_bodies_batch_basic_and_fixed(self):
        batch_size = 2
        state, names = self.create_registered_state(
            batch_size=batch_size, include_connectors=False
        )
        num_update = len(names)
        # New positions within bounds; body_1 is fixed and must not update
        new_positions = torch.randn(batch_size, num_update, 3) * 0.05
        new_positions[:, :, 2] = torch.abs(new_positions[:, :, 2]) + 0.05
        new_rots = torch.zeros(batch_size, num_update, 4)
        new_rots[..., 0] = 1.0

        updated = update_bodies_batch(state, names, new_positions, new_rots)

        # body_0 and body_2 update; body_1 stays unchanged (zeros)
        assert torch.allclose(
            updated.position[:, [0, 2], :], new_positions[:, [0, 2], :]
        )
        assert torch.allclose(updated.quat[:, [0, 2], :], new_rots[:, [0, 2], :])
        assert torch.all(updated.position[:, 1, :] == 0)
        assert torch.all(updated.quat[:, 1, 0] == 1.0)
        assert torch.all(updated.quat[:, 1, 1:] == 0)

    def test_update_bodies_batch_bounds_validation(self):
        batch_size = 1
        state, names = self.create_registered_state(
            batch_size=batch_size, include_connectors=False
        )
        num_update = len(names)
        # Create out-of-bounds z
        positions = torch.zeros(batch_size, num_update, 3)
        positions[..., 2] = 10.0  # way out of bounds
        rotations = torch.zeros(batch_size, num_update, 4)
        rotations[..., 0] = 1.0

        with pytest.raises(AssertionError, match="out of bounds"):
            update_bodies_batch(state, names, positions, rotations)

    def test_update_bodies_batch_connectors_update(self):
        batch_size = 2
        state, names = self.create_registered_state(
            batch_size=batch_size, include_connectors=True
        )
        num_update = len(names)

        # Move all by a known delta; identity rotations
        delta = torch.tensor([0.02, -0.03, 0.04])
        positions = torch.zeros(batch_size, num_update, 3)
        positions += delta
        positions[..., 2] = torch.abs(positions[..., 2]) + 0.05
        rotations = torch.zeros(batch_size, num_update, 4)
        rotations[..., 0] = 1.0

        connector_rel = torch.tensor(
            [
                [float("nan"), float("nan"), float("nan")],
                [0.10, 0.00, 0.05],  # male
                [-0.10, 0.00, 0.05],  # female
            ]
        )

        updated = update_bodies_batch(state, names, positions, rotations, connector_rel)

        # Expected connector positions with identity rotations: pos + rel
        # Names order: [body_0, male, female]; male array contains only male connectors in that order
        expected_male = positions[:, 1:2, :] + connector_rel[1:2, :]
        expected_female = positions[:, 2:3, :] + connector_rel[2:3, :]

        assert torch.allclose(updated.male_connector_positions, expected_male)
        assert torch.allclose(updated.female_connector_positions, expected_female)

    def test_update_bodies_batch_device_handling(self):
        batch_size = 1
        state, names = self.create_registered_state(
            batch_size=batch_size, include_connectors=False
        )
        num_update = len(names)

        positions = torch.randn(batch_size, num_update, 3) * 0.05
        positions[..., 2] = torch.abs(positions[..., 2]) + 0.05
        rotations = torch.zeros(batch_size, num_update, 4)
        rotations[..., 0] = 1.0

        if torch.cuda.is_available():
            positions = positions.to("cuda")
            rotations = rotations.to("cuda")

        updated = update_bodies_batch(state, names, positions, rotations)

        # Verify tensors are on CPU (state.device)
        assert updated.position.device.type == "cpu"
        assert updated.quat.device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
