import trimesh
import genesis as gs
from repairs_components.geometry.connectors.models.europlug import Europlug
from pathlib import Path

gs.init()
europlug = Europlug(0)
obj_path = europlug.save_path_from_name(Path("/workspace/data/"), "europlug_0_male", "obj")
mjcf_path = europlug.save_path_from_name(Path("/workspace/data/"), "europlug_0_male", "xml")
mjcf_text = europlug.get_mjcf(base_dir=Path("/workspace/data/"), male=True)
with open(mjcf_path, "w") as f:
    f.write(mjcf_text)

mesh = trimesh.load(obj_path)
print("mesh volume:", mesh.volume) # mesh is big enough

scene = gs.Scene()
mjcf = gs.morphs.MJCF(file=str(mjcf_path), scale=1)
scene.add_entity(mjcf)


