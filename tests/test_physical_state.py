import pytest
import torch

from repairs_components.logic.physical_state import PhysicalState, register_fasteners_batch
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
    state.count_fasteners_held = torch.zeros(num_bodies, dtype=torch.int8, device=device)
    
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
            single_state.position = torch.rand(num_bodies, 3, device=self.device) * 0.32 - 0.16
            single_state.position[:, 2] = torch.abs(single_state.position[:, 2])
            
            quat = torch.randn(num_bodies, 4, device=self.device)
            single_state.quat = quat / torch.norm(quat, dim=-1, keepdim=True)
            
            single_state.fixed = torch.zeros(num_bodies, dtype=torch.bool, device=self.device)
            single_state.count_fasteners_held = torch.zeros(num_bodies, dtype=torch.int8, device=self.device)
            single_state.part_hole_batch = torch.randint(0, num_bodies, (num_holes,), device=self.device)
        
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
            fastener_init_hole_b
        )
        
        fastener_compound_names = [
            get_fastener_singleton_name(3.0, 10.0),
            get_fastener_singleton_name(4.0, 12.0),
            get_fastener_singleton_name(5.0, 16.0)
        ]
        
        # Call the function
        result_state = register_fasteners_batch(
            physical_state,
            fastener_pos,
            fastener_quat,
            fastener_init_hole_a,
            fastener_init_hole_b,
            fastener_compound_names
        )
        
        # Verify the result
        assert result_state is physical_state  # Should modify in place
        assert result_state.fasteners_pos.shape == (batch_size, num_fasteners, 3)
        assert result_state.fasteners_quat.shape == (batch_size, num_fasteners, 4)
        assert result_state.fasteners_diam.shape == (batch_size, num_fasteners)
        assert result_state.fasteners_length.shape == (batch_size, num_fasteners)
        assert result_state.fasteners_attached_to_hole.shape == (batch_size, num_fasteners, 2)
        assert result_state.fasteners_attached_to_body.shape == (batch_size, num_fasteners, 2)
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
            get_fastener_singleton_name(4.0, 12.0)
        ]
        
        result_state = register_fasteners_batch(
            physical_state,
            fastener_pos,
            fastener_quat,
            fastener_init_hole_a,
            fastener_init_hole_b,
            fastener_compound_names
        )
        
        # Verify fastener parameters are set correctly
        expected_diam_0, expected_length_0 = 3.0, 10.0
        expected_diam_1, expected_length_1 = 4.0, 12.0
        
        assert torch.allclose(result_state.fasteners_diam[0, 0], torch.tensor(expected_diam_0))
        assert torch.allclose(result_state.fasteners_length[0, 0], torch.tensor(expected_length_0))
        assert torch.allclose(result_state.fasteners_diam[0, 1], torch.tensor(expected_diam_1))
        assert torch.allclose(result_state.fasteners_length[0, 1], torch.tensor(expected_length_1))

    def test_register_fasteners_batch_hole_attachment(self):
        """Test that fasteners are correctly attached to holes and bodies."""
        batch_size = 1
        num_fasteners = 2
        physical_state = self.create_batched_physical_state(batch_size=batch_size)
        
        # Set specific hole-to-body mapping for predictable testing
        physical_state.part_hole_batch = torch.tensor([[0, 1, 2, 0, 1, 2]])  # holes map to bodies
        
        fastener_pos = torch.randn(batch_size, num_fasteners, 3)
        fastener_quat = torch.randn(batch_size, num_fasteners, 4)
        fastener_init_hole_a = torch.tensor([[0, 2]])  # holes 0, 2
        fastener_init_hole_b = torch.tensor([[1, 3]])  # holes 1, 3
        fastener_compound_names = [
            get_fastener_singleton_name(3.0, 10.0),
            get_fastener_singleton_name(4.0, 12.0)
        ]
        
        result_state = register_fasteners_batch(
            physical_state,
            fastener_pos,
            fastener_quat,
            fastener_init_hole_a,
            fastener_init_hole_b,
            fastener_compound_names
        )
        
        # Verify hole attachments
        expected_holes = torch.tensor([[[0, 1], [2, 3]]])
        assert torch.equal(result_state.fasteners_attached_to_hole, expected_holes)
        
        # Verify body attachments (based on part_hole_batch mapping)
        expected_bodies = torch.tensor([[[0, 1], [2, 0]]])  # holes 0,1,2,3 -> bodies 0,1,2,0
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
            get_fastener_singleton_name(4.0, 12.0)
        ]
        
        result_state = register_fasteners_batch(
            physical_state,
            fastener_pos,
            fastener_quat,
            fastener_init_hole_a,
            fastener_init_hole_b,
            fastener_compound_names
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
        physical_state.part_hole_batch = torch.tensor([[0, 0, 1, 1, 2, 2]])  # 2 holes per body
        
        fastener_pos = torch.randn(batch_size, num_fasteners, 3)
        fastener_quat = torch.randn(batch_size, num_fasteners, 4)
        # Fastener 0: body 0 to body 1, Fastener 1: body 1 to body 2, Fastener 2: body 0 to body 2
        fastener_init_hole_a = torch.tensor([[0, 2, 1]])  # holes 0, 2, 1 -> bodies 0, 1, 0
        fastener_init_hole_b = torch.tensor([[3, 4, 5]])  # holes 3, 4, 5 -> bodies 1, 2, 2
        fastener_compound_names = [
            get_fastener_singleton_name(3.0, 10.0),
            get_fastener_singleton_name(4.0, 12.0),
            get_fastener_singleton_name(5.0, 16.0)
        ]
        
        result_state = register_fasteners_batch(
            physical_state,
            fastener_pos,
            fastener_quat,
            fastener_init_hole_a,
            fastener_init_hole_b,
            fastener_compound_names
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
                [get_fastener_singleton_name(3.0, 10.0)] * num_fasteners
            )
        
        # Test wrong fastener_quat shape
        with pytest.raises(AssertionError, match="Expected fastener_quat shape"):
            register_fasteners_batch(
                physical_state,
                torch.randn(batch_size, num_fasteners, 3),
                torch.randn(batch_size, num_fasteners, 3),  # Wrong last dimension
                torch.randint(0, 6, (batch_size, num_fasteners)),
                torch.randint(0, 6, (batch_size, num_fasteners)),
                [get_fastener_singleton_name(3.0, 10.0)] * num_fasteners
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
        
        single_state.position = torch.rand(num_bodies, 3, device=self.device) * 0.32 - 0.16
        single_state.position[:, 2] = torch.abs(single_state.position[:, 2])
        
        quat = torch.randn(num_bodies, 4, device=self.device)
        single_state.quat = quat / torch.norm(quat, dim=-1, keepdim=True)
        
        single_state.fixed = torch.zeros(num_bodies, dtype=torch.bool, device=self.device)
        single_state.count_fasteners_held = torch.zeros(num_bodies, dtype=torch.int8, device=self.device)
        single_state.part_hole_batch = torch.randint(0, num_bodies, (num_holes,), device=self.device)
        
        physical_state: PhysicalState = torch.stack([single_state])  # type: ignore
        
        fastener_pos = torch.randn(batch_size, num_fasteners, 3)
        fastener_quat = torch.randn(batch_size, num_fasteners, 4)
        fastener_compound_names = [
            get_fastener_singleton_name(3.0, 10.0),
            get_fastener_singleton_name(4.0, 12.0)
        ]
        
        # Test out-of-range hole indices (now hole 4 is out of range since we only have 4 holes: 0,1,2,3)
        with pytest.raises(AssertionError, match="fastener_init_hole_a are out of range"):
            register_fasteners_batch(
                physical_state,
                fastener_pos,
                fastener_quat,
                torch.tensor([[0, 4]]),  # hole 4 is out of range (max is 3)
                torch.tensor([[1, 2]]),
                fastener_compound_names
            )
        
        with pytest.raises(AssertionError, match="fastener_init_hole_b are out of range"):
            register_fasteners_batch(
                physical_state,
                fastener_pos,
                fastener_quat,
                torch.tensor([[0, 1]]),
                torch.tensor([[2, 5]]),  # hole 5 is out of range (max is 3)
                fastener_compound_names
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
            get_fastener_singleton_name(4.0, 12.0)
        ]
        
        # Test same hole indices (should fail)
        with pytest.raises(AssertionError, match="Fastener init holes must be different or empty"):
            register_fasteners_batch(
                physical_state,
                fastener_pos,
                fastener_quat,
                torch.tensor([[0, 1]]),
                torch.tensor([[0, 2]]),  # First fastener has same hole for both ends
                fastener_compound_names
            )
        
        # Test both holes as -1 (should pass)
        result_state = register_fasteners_batch(
            physical_state,
            fastener_pos,
            fastener_quat,
            torch.tensor([[-1, 0]]),
            torch.tensor([[-1, 1]]),  # First fastener unattached (both -1)
            fastener_compound_names
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
        single_state.count_fasteners_held = torch.zeros(1, dtype=torch.int8, device=self.device)
        
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
                [get_fastener_singleton_name(3.0, 10.0)]
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
        
        with pytest.raises(AssertionError, match="Fasteners can't be registered as bodies"):
            register_fasteners_batch(
                physical_state,
                fastener_pos,
                fastener_quat,
                fastener_init_hole_a,
                fastener_init_hole_b,
                [fastener_name]
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
            fastener_compound_names
        )
        
        # Verify tensors are on the correct device
        assert result_state.fasteners_pos.device == target_device
        assert result_state.fasteners_quat.device == target_device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
