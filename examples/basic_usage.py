"""Basic example demonstrating the use of repair components."""

import numpy as np

# import mujoco # TODO remove mujoco!
from pathlib import Path

from repairs_components import Screw, BasicSocket, LockingSocket


def main():
    """Run a simple demonstration of the repair components."""
    # Create components
    screw = Screw(thread_pitch=0.5, length=10.0, name="test_screw")
    basic_socket = BasicSocket(size=8.0, name="basic_socket")
    locking_socket = LockingSocket(
        size=10.0, requires_release=True, name="locking_socket"
    )

    # Print initial states
    print("Initial states:")
    print(f"Screw: {screw.state}")
    print(f"Basic Socket: {basic_socket.state}")
    print(f"Locking Socket: {locking_socket.state}")

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

    # Create data
    data = mujoco.MjData(model)

    # Attach components to the model
    screw.attach_to_model(model, data)
    basic_socket.attach_to_model(model, data)
    locking_socket.attach_to_model(model, data)

    # Simulate
    print("\nSimulating...")
    for i in range(100):
        # In a real implementation, you would apply forces/torques based on actions
        if i == 10:
            print("\nFastening screw...")
            screw.fasten(np.pi / 2)  # Rotate 90 degrees
            print(f"Screw position after fastening: {screw._position:.2f} mm")

            print("\nConnecting basic socket...")
            if basic_socket.connect(force=15.0):
                print("  Successfully connected basic socket")

            print("\nConnecting locking socket...")
            if locking_socket.connect(force=15.0):
                print("  Successfully connected locking socket")

            print("\nTrying to disconnect locking socket (should fail)...")
            if not locking_socket.disconnect():
                print(
                    "  Failed to disconnect locking socket (as expected, release not activated)"
                )

            print("\nActivating release and disconnecting locking socket...")
            if locking_socket.activate_release(force=6.0):  # Above release force
                print("  Release mechanism activated")
                if locking_socket.disconnect():
                    print("  Successfully disconnected locking socket")

        # Step simulation
        mujoco.mj_step(model, data)

        # Update components
        screw.step()
        basic_socket.step()
        locking_socket.step()

    # Print final states
    print("\nFinal states:")
    print(f"Screw: {screw.state}")
    print(f"Basic Socket: {basic_socket.state}")
    print(f"Locking Socket: {locking_socket.state}")


if __name__ == "__main__":
    main()
