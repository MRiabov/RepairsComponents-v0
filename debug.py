if __name__ == "__main__":
    from src.geometry.connectors.models.europlug import Europlug
    from src.processing.voxel_export import export_voxel_grid
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
    from build123d import Color

    # Create Europlug parts
    part_male, connector_def_male, part_female, connector_def_female = (
        Europlug().bd_geometry((0, 50, 0), (0, -50, 0))
    )

    # Set part labels with types
    part_male.label = "male_plug@connector"
    part_female.label = "female_plug@connector"

    # Export to voxel grid
    voxel_grid, metadata = export_voxel_grid([part_male, part_female], voxel_size=1.0)

    print(f"Voxel grid shape: {voxel_grid.shape}")
    print(f"Part types: {metadata['part_types']}")

    # Visualize the voxel grid
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Get the coordinates of the filled voxels
    filled = np.argwhere(voxel_grid > 0)

    if len(filled) > 0:
        # Get colors for each voxel
        print("starting to get colors")
        colors = []
        for idx in filled:
            part_idx = voxel_grid[tuple(idx)]
            part_type = metadata["part_types"][part_idx]
            color = metadata["colors"].get(part_type, [1, 0, 0, 0.5])
            colors.append(color[:4])  # Ensure we have RGBA

        # Plot the voxels
        ax.voxels(
            filled=voxel_grid > 0,
            facecolors=np.zeros((*voxel_grid.shape, 4)),  # Transparent by default
            edgecolor="k",
            linewidth=0.1,
        )

        print("voxelization ready")

        # Plot each point individually with its color
        for (x, y, z), color in zip(filled, colors):
            ax.scatter(
                x,
                y,
                z,
                c=[color],
                marker="o",
                s=100,  # Adjust size as needed
                alpha=color[3] if len(color) > 3 else 1.0,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Voxelized Europlug Connectors")

    plt.tight_layout()
    plt.show()
