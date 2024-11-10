import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
import numpy as np
from scipy.spatial import cKDTree
import numpy.ma as ma  # For masked arrays

def calculate_combined_risks(Z_grid, non_nan_indices, max_height_diff=0.5, max_slope_degrees=30.0, radius=0.3):
    """
    Calculate both step and slope risks based on height differences and gradients between neighboring cells,
    considering only neighbors within a specified radius for slope risk.
    """
    # Initialize grids for step and slope risks
    step_risk_grid = np.full_like(Z_grid, np.nan)
    slope_risk_grid = np.full_like(Z_grid, np.nan)

    # Define offsets for 8 neighbors for step risk
    all_neighbors_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    # Convert max slope degrees to radians for slope normalization
    max_slope_radians = np.deg2rad(max_slope_degrees)

    # Filter offsets for slope risk to include only those within the radius
    filtered_slope_offsets = [
        (dx, dy) for dx, dy in all_neighbors_offsets
        if np.sqrt((dx * radius) ** 2 + (dy * radius) ** 2) <= radius
    ]

    for x, y in non_nan_indices:
        z_center = Z_grid[x, y]
        if np.isnan(z_center):
            continue

        # Initialize lists to store height differences and slopes
        height_diffs = []
        slopes = []

        # Calculate height differences with 8 neighbors for step risk
        for dx, dy in all_neighbors_offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < Z_grid.shape[0] and 0 <= ny < Z_grid.shape[1]:
                z_neighbor = Z_grid[nx, ny]
                if not np.isnan(z_neighbor):
                    height_diffs.append(abs(z_neighbor - z_center))

        # Calculate slopes with filtered neighbors for slope risk
        for dx, dy in filtered_slope_offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < Z_grid.shape[0] and 0 <= ny < Z_grid.shape[1]:
                z_neighbor = Z_grid[nx, ny]
                if not np.isnan(z_neighbor):
                    # Calculate XY distance
                    xy_distance = np.sqrt((dx * radius) ** 2 + (dy * radius) ** 2)
                    # Calculate the slope angle
                    slope_angle = np.arctan2(abs(z_neighbor - z_center), xy_distance)
                    slopes.append(slope_angle)

        # Calculate step risk by normalizing max height difference
        if height_diffs:
            max_diff = np.max(height_diffs)
            step_risk = min(max_diff / max_height_diff, 1.0)
            step_risk_grid[x, y] = step_risk
        else:
            step_risk_grid[x, y] = 0

        # Calculate slope risk by normalizing max slope angle
        if slopes:
            max_slope = max(slopes)
            slope_risk = min(max_slope / max_slope_radians, 1.0)
            slope_risk_grid[x, y] = slope_risk
        else:
            slope_risk_grid[x, y] = 0

    return step_risk_grid, slope_risk_grid


# Load the PLY point cloud files
ground_point_cloud = o3d.io.read_point_cloud('ground_point_cloud.ply')
obstacle_point_cloud = o3d.io.read_point_cloud('obstacle_point_cloud.ply')

# Convert the point clouds to numpy arrays
ground_points = np.asarray(ground_point_cloud.points)
obstacle_points = np.asarray(obstacle_point_cloud.points)

# Extract X, Y, Z for ground points
x_vals_ground = ground_points[:, 0]
y_vals_ground = ground_points[:, 1]
z_vals_ground = ground_points[:, 2]

# Define the grid resolution
grid_resolution = 0.10

# Create grid edges for X and Y based on the range of ground points
x_edges = np.arange(min(x_vals_ground), max(x_vals_ground) + grid_resolution, grid_resolution)
y_edges = np.arange(min(y_vals_ground), max(y_vals_ground) + grid_resolution, grid_resolution)

# Create meshgrid for X and Y (for ground)
x_mid = (x_edges[:-1] + x_edges[1:]) / 2  # Midpoints of X bins
y_mid = (y_edges[:-1] + y_edges[1:]) / 2  # Midpoints of Y bins
X, Y = np.meshgrid(x_mid, y_mid)

# Initialize an empty Z grid for ground points
Z_ground = np.full((len(x_mid), len(y_mid)), np.nan)

# Fill the Z grid for ground points
for i in range(len(x_vals_ground)):
    x_idx = np.digitize(x_vals_ground[i], x_edges) - 1
    y_idx = np.digitize(y_vals_ground[i], y_edges) - 1
    if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
        Z_ground[x_idx, y_idx] = z_vals_ground[i]

# Get the list of non-NaN indices in Z_ground
non_nan_indices = np.argwhere(~np.isnan(Z_ground))


# Calculate the combined step and slope risk grids
step_risk_grid, slope_risk_grid = calculate_combined_risks(Z_ground, non_nan_indices, max_height_diff=0.10, max_slope_degrees=30.0, radius=0.5)

# # Merge the two risk grids to get the total risk grid
total_risk_grid = np.nanmean([step_risk_grid, slope_risk_grid], axis=0)

# Create a custom colormap from gray to yellow to red
colors = [(0.5, 0.5, 0.5), (1, 1, 0), (1, 0, 0)]  # Gray → Yellow → Red
cmap = LinearSegmentedColormap.from_list("gray_yellow_red", colors)

# Plot the total risk grid
fig, ax = plt.subplots()
c = ax.pcolormesh(X, Y, total_risk_grid.T, shading='auto', cmap=cmap, vmin=0, vmax=1)
fig.colorbar(c, ax=ax, label='Total Risk (0-1)', ticks=[0, 0.25, 0.5, 0.75, 1])

# Set axis labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Total Risk Grid')

plt.show()