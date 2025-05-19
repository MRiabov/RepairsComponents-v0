"""Example demonstrating the Button component in a simple MuJoCo environment."""
import numpy as np
import mujoco
from mujoco import viewer as mujoco_viewer

# Simple model without any components for testing
MODEL_XML = """
<mujoco>
    <option timestep="0.01"/>
    
    <worldbody>
        <!-- Light and camera -->
        <light name="light" pos="0 0 4"/>
        <camera name="fixed" pos="0 -2 1.5" xyaxes="1 0 0 0 0 1"/>
        
        <!-- Floor -->
        <geom name="floor" type="plane" size="2 2 0.1" rgba=".9 .9 .9 1" pos="0 0 0"/>
        
        <!-- Simple box for testing -->
        <body name="box" pos="0 0 0.5">
            <freejoint/>
            <geom type="box" size="0.2 0.2 0.2" rgba="0.2 0.6 0.8 1"/>
        </body>
    </worldbody>
</mujoco>
"""

def main():
    """Run a simple MuJoCo simulation."""
    print("Creating model...")
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    print("Creating data...")
    data = mujoco.MjData(model)
    print("Model and data created successfully")
    
    # Launch the viewer
    with mujoco_viewer.launch_passive(model, data) as viewer:
        # Set camera
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 0
        viewer.cam.elevation = -20
        
        # Main simulation loop
        while viewer.is_running():
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Sync the viewer
            viewer.sync()

if __name__ == "__main__":
    main()
