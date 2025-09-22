from matplotlib import pyplot as plt
import numpy as np

def auxiliary_heatmap(positions, size=14):
    """
    positions must contain at least 3 triplets of (x, y, v)
    size refers to the number of cell centers along each dimension
    """
    heatmap = np.zeros((size, size))
    positions = positions[-3:]
    # We have to find the index of the last 0
    i = 0
    for (x, y, v) in positions[::-1]:
        if v < 0.5:
            break
        i += 1
    if i <= 1:
        return heatmap
    elif i == 2: # We have two points to work with, we use linear interpolation
        (x_2, y_2, vis) = positions[1]
        (x_3, y_3, vis) = positions[2]
        x_pred = 2 * x_3 - x_2
        y_pred = 2 * y_3 - y_2
    else: # We have three visible points immediately before, we can perform quadratic interpolation
        (x_1, y_1, vis) = positions[0]
        (x_2, y_2, vis) = positions[1]
        (x_3, y_3, vis) = positions[2]
        x_pred = x_1 - 3*x_2 + 3*x_3
        y_pred = y_1 - 3*y_2 + 3*y_3
    print(x_pred, y_pred)
    # We check that the points aren't too close together and that the prediction is in [0, 1] x [0, 1]
    if np.linalg.norm(np.array([x_2 - x_3, y_2 - y_3])) + np.linalg.norm(np.array([x_3 - x_pred, y_3 - y_pred])) <= \
        0.02 or not (0 <= x_pred <= 1 and 0 <= y_pred <= 1):
        return heatmap

    sigma = 0.5 / size
    cell_centers = []
    for row in range(size):
        for col in range(size):
            cell_center = np.array([0.5/size + col / size, 0.5/size + row / size]) # Expressed in xy coordinates
            cell_centers.append(cell_center)
            dist = np.linalg.norm(cell_center - np.array([x_pred, y_pred]))
            heatmap[row, col] = np.exp(-dist**2 / (2*sigma**2))

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    # Show the heatmap over [0,1]x[0,1]
    cax = ax.imshow(
        heatmap,
        origin="lower",
        extent=[0, 1, 0, 1],
        interpolation="nearest",
        aspect="equal",
        cmap="viridis"
    )

    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Heatmap intensity")

    # Draw a size×size grid
    # ax.set_xticks(np.linspace(0, 1, size + 1))
    # ax.set_yticks(np.linspace(0, 1, size + 1))
    # Minor ticks (for the full grid, including gridlines every cell)
    ax.set_xticks(np.linspace(0, 1, size + 1), minor=True)
    ax.set_yticks(np.linspace(0, 1, size + 1), minor=True)

    # Major ticks (labels every 2 squares)
    ax.set_xticks(np.linspace(0, 1, size // 2 + 1))  # Every 2 out of 14 = 8 labels
    ax.set_yticks(np.linspace(0, 1, size // 2 + 1))

    # Label formatting
    ax.set_xticklabels([f"{x:.2f}" for x in np.linspace(0, 1, size // 2 + 1)])
    ax.set_yticklabels([f"{y:.3f}" for y in np.linspace(0, 1, size // 2 + 1)])
    ax.grid(which="both", linewidth=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{size}×{size} grid with past points and predicted point")
    for (y, x) in cell_centers:
        ax.scatter([x], [y], c="black", marker="o", s=5)

    for j, (x, y, v) in enumerate(positions[::-1], start=1):
        ax.scatter([x], [y], s=60, label=f"point -{j}")
    ax.scatter([x_pred], [y_pred], s=80, marker="X", label="predicted")
    # Clean legend (dedupe)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right")
    plt.show()

    return heatmap

    # Plot the last three past positions if visible and within [0,1]×[0,1]


    # Plot the prediction if valid



positions = np.array([[0,0,0],
                      [0.3,0.3,1],
                      [0.4,0.5,1],
                      [0.5,0.56,1]])
print(np.round(auxiliary_heatmap(positions), 2))

