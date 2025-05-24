from build123d import Pos
from src.geometry.electrical.controls.controls import Button, Switch
from src.geometry.fasteners import Fastener
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from src.processing.voxel_export import PART_TYPE_COLORS
from src.geometry.connectors.models.europlug import Europlug
import ocp_vscode
from src.processing.voxel_export import export_voxel_grid


if __name__ == "__main__":
    europlug_male, connector_def_male, europlug_female, connector_def_female = (
        Europlug().bd_geometry((0, 50, 0), (0, -50, 0))
    )
    screw = Fastener("a", "b", name="screw1").bd_geometry().move(Pos(0, 120, 0))
    button = Button("button1").bd_geometry().move(Pos(0, 0, 75))
    switch = Switch("switch1").bd_geometry().move(Pos(0, 50, 75))
    ocp_vscode.show(
        europlug_male,
        connector_def_male,
        europlug_female,
        connector_def_female,
        screw,
        button,
        switch,
    )

    print("executed")
    padded, label_to_part_type = export_voxel_grid(
        (
            europlug_male,
            connector_def_male,
            europlug_female,
            connector_def_female,
            screw,
            button,
            switch,
        ),
        1,
    )
    # NOTE: Compound(children=[...]) preserves labels, colors and constraints, while Compound(objs) does not
    # you can use Compound after all.

    # 3D voxel visualization

    # Prepare color list (order: 0, 1, 2, 3, 4)
    color_list = [
        PART_TYPE_COLORS[label_to_part_type.get(i, "default")]
        for i in range(max(label_to_part_type.keys()) + 1)
    ]
    cmap = ListedColormap([c[:3] for c in color_list], name="part_types")
    alpha = np.array([c[3] for c in color_list])

    # Create mask for non-background
    filled = padded > 0
    x, y, z = np.nonzero(filled)
    values = padded[x, y, z]

    if x.size == 0:
        print("No voxels to plot (no non-background labels found).")
    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        # Plot voxels
        ax.scatter(
            x,
            y,
            z,
            c=[cmap.colors[v] for v in values],
            alpha=alpha[values],
            marker="s",
            s=6,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Voxelized Parts (3D)")
        ax.set_box_aspect(padded.shape)

        plt.tight_layout()
        plt.show()
