---
trigger: always_on
---

When rendering, you can not use genesis `show_render=True` because it doesn't work with my SSH setup. Render video using programmed camera instead. Simplest demo:
```
cam.start_recording()
import numpy as np

for i in range(120):
    scene.step()
    cam.set_pose(
        pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
        lookat = (0, 0, 0.5),
    )
    cam.render()
cam.stop_recording(save_to_filename='/workspace/RepairsComponents-v0/video.mp4', fps=60)
```