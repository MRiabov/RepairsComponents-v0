import matplotlib.pyplot as plt

if __name__ == "__main__":
    from src.geometry.connectors.models.europlug import Europlug
    import ocp_vscode
    from src.processing.voxel_export import export_voxel_grid

    part_male, connector_def_male, part_female, connector_def_female = (
        Europlug().bd_geometry((0, 50, 0), (0, -50, 0))
    )

    print("executed")
    # ocp_vscode.show(part_male, connector_def_male, part_female, connector_def_female)
    padded, label_to_part_type = export_voxel_grid(
        (part_male, connector_def_male, part_female, connector_def_female), 1
    )

    # 3D voxel visualization
    import numpy as np
    from matplotlib.colors import ListedColormap
    from src.processing.voxel_export import PART_TYPE_COLORS

    # Prepare color list (order: 0, 1, 2, 3, 4)
    color_list = [PART_TYPE_COLORS[label_to_part_type.get(i, 'default')] for i in range(max(label_to_part_type.keys())+1)]
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
        ax = fig.add_subplot(111, projection='3d')
        # Plot voxels
        ax.scatter(x, y, z, c=[cmap.colors[v] for v in values], alpha=alpha[values], marker='s', s=6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Voxelized Parts (3D)')
        plt.tight_layout()
        plt.show()
