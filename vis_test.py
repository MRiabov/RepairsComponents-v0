import mujoco
import mujoco_viewer

model = mujoco.MjModel.from_xml_string("""
<mujoco>
    <option timestep="0.01"/>
    <worldbody>
        <light name="light" pos="0 0 4"/>
        <geom name="floor" type="plane" size="10 10 0.1" rgba=".9 .9 .9 1"/>
        <body name="box" pos="0 0 1">
            <freejoint/>
            <geom type="box" size="0.5 0.5 0.5" rgba="0.2 0.6 0.8 1"/>
        </body>
    </worldbody>
</mujoco>
""")
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
for _ in range(10000):
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

# close
viewer.close()


viewer = mujoco_viewer.MujocoViewer(model, data)
