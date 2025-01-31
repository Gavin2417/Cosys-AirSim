import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load the PLY point cloud file
point_cloud = o3d.io.read_point_cloud('grid_point_cloud.ply')

# Convert the point cloud to a numpy array
points = np.asarray(point_cloud.points)

# Extract X, Y, Z coordinates
x_vals = points[:, 0]
y_vals = points[:, 1]
z_vals = points[:, 2]

# Recreate the points array to be Nx3
points_xyz = np.vstack((x_vals, y_vals, z_vals)).T

def calculate_step_risk(points, grid_resolution=0.1, max_height_diff=0.5):
    """
    Calculate step risk based on height difference between adjacent cells in the grid map.
    Also considers points that are significantly lower than a baseline ground level.
    
    Args:
    - points: Nx3 numpy array of LiDAR points (X, Y, Z)
    - grid_resolution: The grid resolution used for binning points.
    - max_height_diff: The maximum height difference to be considered for risk (e.g., 0.5 meters).
    - ground_threshold: The threshold for ground detection (e.g., -0.55 meters).
    
    Returns:
    - step_risk: Nx1 array of step risks for each point (0 to 1 based on height differences).
    """
    
    # Create grid edges for X and Y based on the range of your X and Y values
    x_edges = np.arange(min(x_vals), max(x_vals) + grid_resolution, grid_resolution)
    y_edges = np.arange(min(y_vals), max(y_vals) + grid_resolution, grid_resolution)

    # Create meshgrid for X and Y
    x_mid = (x_edges[:-1] + x_edges[1:]) / 2  # Midpoints of X bins
    y_mid = (y_edges[:-1] + y_edges[1:]) / 2  # Midpoints of Y bins
    X, Y = np.meshgrid(x_mid, y_mid)

    # Initialize an empty Z grid
    Z_grid = np.full((len(x_mid), len(y_mid)), np.nan)

    # Fill the Z grid with the height values
    for i in range(len(x_vals)):
        x_idx = np.digitize(x_vals[i], x_edges) - 1  # Find the bin index for the x value
        y_idx = np.digitize(y_vals[i], y_edges) - 1  # Find the bin index for the y value

        if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
            Z_grid[x_idx, y_idx] = z_vals[i]

    # Initialize a step risk grid
    step_risk_grid = np.full_like(Z_grid, np.nan)

    # Calculate step risk based on height differences between neighboring cells
    for x in range(1, len(x_mid) - 1):
        for y in range(1, len(y_mid) - 1):
            # Get the current cell height
            z_center = Z_grid[x, y]
            if np.isnan(z_center):
                continue

            # Get neighboring heights (north, south, west, east)
            neighbors = [
                Z_grid[x-1, y-1], Z_grid[x-1, y], Z_grid[x-1, y+1],
                Z_grid[x, y-1], Z_grid[x, y], Z_grid[x, y+1],
                Z_grid[x+1, y-1], Z_grid[x+1, y], Z_grid[x+1, y+1]
            ]
            neighbors = [z for z in neighbors if not np.isnan(z)]  # Ignore NaNs

            if neighbors:
                # Calculate the height difference with the neighbors
                max_diff = np.max(np.abs(np.array(neighbors) - z_center))
                # max_diff = np.mean(np.abs(np.array(neighbors) - z_center))

                # Normalize the risk based on the max height difference
                step_risk = min(max_diff / max_height_diff, 1.0)  # Scale to 0-1
            else:
                step_risk = 0  # If no neighbors, assume no risk

                
            step_risk_grid[x, y] = step_risk

    return step_risk_grid, X, Y

# Calculate step risk
step_risk_grid, X, Y = calculate_step_risk(points_xyz, grid_resolution=0.1, max_height_diff=0.12)

# Create a custom colormap from gray to yellow to red
colors = [(0.5, 0.5, 0.5), (1, 1, 0), (1, 0, 0)]  # Gray → Yellow → Red
cmap = LinearSegmentedColormap.from_list("gray_yellow_red", colors)

# Plot the grid using pcolormesh with step risk
fig, ax = plt.subplots()
c = ax.pcolormesh(X, Y, step_risk_grid.T, shading='auto', cmap=cmap, vmin=0, vmax=1)

# Add a color bar to indicate the step risk values
fig.colorbar(c, ax=ax, label='Step Risk (0-1)', ticks=[0, 0.25, 0.5, 0.75, 1])

# Set axis labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('2D Grid Cell Plot of Step Risk from PLY File')

# access the values of the step risk 

for i in range(len(step_risk_grid)):
    for j in range(len(step_risk_grid[i])):
        if step_risk_grid[i][j] > 0.5:
            print(f"Step risk at ({X[i][j]}, {Y[i][j]}) is {step_risk_grid[i][j]}")
plt.show()
