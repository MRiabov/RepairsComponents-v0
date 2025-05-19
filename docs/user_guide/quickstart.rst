.. _quickstart:

Quickstart Guide
===============

This guide will walk you through creating a simple simulation using RepairsComponents.

Basic Usage
----------

Let's create a simple simulation with a screw and a button:

.. code-block:: python

    import mujoco
    import numpy as np
    from mujoco import viewer
    from repairs_components import Screw, Button

    def main():
        # Create a simple MuJoCo model
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
        data = mujoco.MjData(model)

        # Create components
        screw = Screw(thread_pitch=0.5, length=10.0, name="test_screw")
        button = Button(press_force=2.0, name="test_button")

        # Attach components to the model
        for component in [screw, button]:
            component.attach_to_model(model, data)

        # Simulation loop
        with viewer.launch_passive(model, data) as v:
            v.cam.distance = 3.0
            v.cam.azimuth = 0
            v.cam.elevation = -20

            for step in range(1000):
                # Fasten the screw
                screw.fasten(0.01)
                
                # Press the button every 100 steps
                if step % 100 == 0:
                    button.press(force=3.0)
                
                # Step the simulation
                mujoco.mj_step(model, data)
                
                # Update components
                screw.step()
                button.step()
                
                # Sync the viewer
                v.sync()

    if __name__ == "__main__":
        main()

Component Interaction
--------------------

Components can interact with each other through callbacks:

.. code-block:: python

    from repairs_components import LockingSocket, Button

    # Create a locking socket and a button
    socket = LockingSocket(size=10.0, requires_release=True, name="power_socket")
    
    def on_button_press(state):
        if state:  # Button pressed
            print("Button pressed - releasing socket")
            socket.activate_release(force=6.0)
            socket.disconnect()

    # Create a button that releases the socket when pressed
    button = Button(
        on_press=on_button_press,
        press_force=2.0,
        name="release_button"
    )

Simulation Loop
--------------

Here's a more detailed example of a simulation loop with error handling:

.. code-block:: python

    import time

    def simulate(model, data, components, steps=1000, step_time=0.01):
        """Run a simulation with the given components."""
        try:
            for step in range(steps):
                # Apply forces, update controls, etc.
                
                # Step the simulation
                mujoco.mj_step(model, data)
                
                # Update components
                for component in components:
                    component.step()
                
                # Optional: Add a small delay for visualization
                time.sleep(step_time)
                
        except KeyboardInterrupt:
            print("Simulation stopped by user")
        except Exception as e:
            print(f"Error during simulation: {e}")
        finally:
            # Cleanup code here
            pass

Visualization Tips
-----------------

For better visualization in the MuJoCo viewer:

1. **Camera Control**:
   - Right-click and drag to rotate
   - Scroll to zoom
   - Shift + right-click and drag to pan

2. **Visualization Options**:
   - Press `t` to toggle transparency
   - Press `r` to reset the view
   - Press `v` to toggle visualization options

3. **Debug Visualization**:
   - Press `d` to show contact points and forces
   - Press `c` to show constraints

Next Steps
----------

- Learn more about the available components in the :ref:`components` guide
- Check out the :ref:`examples` for more complex use cases
- Explore the :ref:`API reference <api_reference>` for detailed documentation
