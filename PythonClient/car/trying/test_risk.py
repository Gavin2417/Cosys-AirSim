import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
import numpy as np
from scipy.spatial import cKDTree
import numpy.ma as ma  # For masked arrays

def calculate_step_risk(Z_grid, non_nan_indices, max_height_diff=0.5):
    """
    Calculate step risk based on height differences between neighboring cells.
    """
    step_risk_grid = np.full_like(Z_grid, np.nan)

    # Define offsets for the 8 neighbors around each cell
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    # Iterate over the non-NaN indices
    for x, y in non_nan_indices:
        z_center = Z_grid[x, y]
        if np.isnan(z_center):
            continue

        neighbors = []
        for dx, dy in neighbor_offsets:
            nx, ny = x + dx, y + dy
            # Check if neighbor is within bounds
            if 0 <= nx < Z_grid.shape[0] and 0 <= ny < Z_grid.shape[1]:
                z_neighbor = Z_grid[nx, ny]
                if not np.isnan(z_neighbor):
                    neighbors.append(z_neighbor)

        if neighbors:
            # Calculate the height difference with the neighbors
            max_diff = np.max(np.abs(np.array(neighbors) - z_center))
            # Normalize the risk based on the max height difference
            step_risk = min(max_diff / max_height_diff, 1.0)  # Scale to 0-1
        else:
            step_risk = 0  # If no neighbors, assume no risk

        step_risk_grid[x, y] = step_risk

    return step_risk_grid

def calculate_slope_risk(Z_grid, non_nan_indices, max_slope_degrees=30.0, radius=0.3):
    """
    Calculate slope risk based on gradient between cells.
    """
    slope_risk_grid = np.full_like(Z_grid, np.nan)

    # Define the 4 adjacent neighbors (up, down, left, right) for slope calculation
    neighbor_offsets = [
        (-1, 0), (1, 0), (0, -1), (0, 1)
    ]

    # Convert max slope degrees to radians for comparison
    max_slope_radians = np.deg2rad(max_slope_degrees)

    for x, y in non_nan_indices:
        z_center = Z_grid[x, y]
        if np.isnan(z_center):
            continue

        slopes = []
        for dx, dy in neighbor_offsets:
            nx, ny = x + dx, y + dy
            # Check if neighbor is within bounds
            if 0 <= nx < Z_grid.shape[0] and 0 <= ny < Z_grid.shape[1]:
                z_neighbor = Z_grid[nx, ny]
                if not np.isnan(z_neighbor):
                    # Calculate the distance in the XY plane
                    xy_distance = np.sqrt((dx * radius) ** 2 + (dy * radius) ** 2)
                    # Calculate the height difference
                    z_diff = z_neighbor - z_center
                    # Calculate the slope (angle in radians)
                    slope_angle = np.arctan2(abs(z_diff), xy_distance)
                    slopes.append(slope_angle)

        if slopes:
            max_slope = max(slopes)
            # Normalize the risk based on the max slope angle
            slope_risk = min(max_slope / max_slope_radians, 1.0)  # Scale to 0-1
        else:
            slope_risk = 0  # If no valid neighbors, assume no risk

        slope_risk_grid[x, y] = slope_risk

    return slope_risk_grid


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

# Calculate the step risk grid
step_risk_grid = calculate_step_risk(Z_ground, non_nan_indices, max_height_diff=0.12)
slope_risk_grid = calculate_slope_risk(Z_ground, non_nan_indices, max_slope_degrees=22.0, radius=0.3)

#Merge the two risk grids to get the total risk grid
total_risk_grid = np.mean([step_risk_grid, slope_risk_grid], axis=0)

# Plot the step risk grid
fig, ax = plt.subplots()
# Create a custom colormap from gray to yellow to red
colors = [(0.5, 0.5, 0.5), (1, 1, 0), (1, 0, 0)]  # Gray → Yellow → Red
cmap = LinearSegmentedColormap.from_list("gray_yellow_red", colors)

# Plot the grid using pcolormesh with step risk
c = ax.pcolormesh(X, Y, slope_risk_grid.T, shading='auto', cmap=cmap, vmin=0, vmax=1)
fig.colorbar(c, ax=ax, label='Step Risk (0-1)', ticks=[0, 0.25, 0.5, 0.75, 1])

# Set axis labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Step Risk Grid')

plt.show()
