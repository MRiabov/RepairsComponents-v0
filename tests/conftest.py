"""Pytest configuration and fixtures."""
import pytest
import mujoco


@pytest.fixture
def basic_mujoco_model():
    """Create a basic MuJoCo model for testing."""
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
        <option timestep="0.01"/>
        <worldbody>
            <light name="light" pos="0 0 4"/>
            <camera name="fixed" pos="0 -1 0.5" xyaxes="1 0 0 0 0 1"/>
            <geom name="floor" type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
        </worldbody>
    </mujoco>
    """)
    return model


@pytest.fixture
def basic_mujoco_data(basic_mujoco_model):
    """Create MuJoCo data for the test model."""
    return mujoco.MjData(basic_mujoco_model)
