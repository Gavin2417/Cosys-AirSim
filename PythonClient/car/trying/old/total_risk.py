import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import LinearSegmentedColormap
from function2 import calculate_combined_risks, compute_cvar_cellwise
from scipy.spatial import cKDTree
import numpy.ma as ma

def interpolate_in_radius(grid, radius):
    """
    Interpolates NaN values in a grid using valid points within a specified radius.
    
    Parameters:
        grid (ndarray): 2D grid with NaN values to interpolate.
        radius (float): Radius within which to search for valid points.
    
    Returns:
        ndarray: Grid with interpolated values.
    """
    # Get valid (non-NaN) points
    valid_points = ~np.isnan(grid)
    valid_coords = np.column_stack(np.where(valid_points))
    valid_values = grid[valid_points]

    # Create KDTree for efficient neighbor search
    tree = cKDTree(valid_coords)

    # Get NaN points
    nan_coords = np.column_stack(np.where(np.isnan(grid)))

    # Iterate through each NaN point
    for idx, coord in enumerate(nan_coords):
        # Find all valid points within the radius
        neighbors = tree.query_ball_point(coord, radius)

        # If there are neighbors, compute a weighted average
        if neighbors:
            weights = []
            weighted_values = []
            for neighbor_idx in neighbors:
                neighbor_coord = valid_coords[neighbor_idx]
                value = valid_values[neighbor_idx]

                # Compute weight based on inverse distance
                distance = np.linalg.norm(coord - neighbor_coord)
                weight = 1 / (distance + 1e-6)  # Add small epsilon to avoid division by zero
                weights.append(weight)
                weighted_values.append(weight * value)

            # Interpolated value is weighted average
            grid[coord[0], coord[1]] = np.sum(weighted_values) / np.sum(weights)

    return grid

# Load the PLY point cloud file
point_cloud = o3d.io.read_point_cloud('grid_point_cloud.ply')
obs_point_cloud = o3d.io.read_point_cloud('obstacle_point_cloud.ply')

# Convert the point cloud to a numpy array
points = np.asarray(point_cloud.points)
obs_point_cloud = np.asarray(obs_point_cloud.points)

# Extract X, Y, Z for ground points
ground_x_vals = points[:, 0]
ground_y_vals = points[:, 1]
ground_z_vals = points[:, 2]

# (Optional) If you have obs_point_cloud defined elsewhere:
obstacle_x_vals = obs_point_cloud[:, 0]
obstacle_y_vals = obs_point_cloud[:, 1]
obstacle_z_vals = obs_point_cloud[:, 2]

# Define the grid resolution
grid_resolution = 0.1

# Create grid edges for X and Y based on the range of ground points
x_edges = np.arange(min(ground_x_vals), max(ground_x_vals) + grid_resolution, grid_resolution)
y_edges = np.arange(min(ground_y_vals), max(ground_y_vals) + grid_resolution, grid_resolution)

# Create meshgrid for X and Y (for ground)
x_mid = (x_edges[:-1] + x_edges[1:]) / 2  # Midpoints of X bins
y_mid = (y_edges[:-1] + y_edges[1:]) / 2  # Midpoints of Y bins
X, Y = np.meshgrid(x_mid, y_mid)

# Initialize an empty Z grid for ground points
Z_ground = np.full((len(x_mid), len(y_mid)), np.nan)

# Fill the Z grid for ground points
for i in range(len(ground_x_vals)):
    x_idx = np.digitize(ground_x_vals[i], x_edges) - 1
    y_idx = np.digitize(ground_y_vals[i], y_edges) - 1
    if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
        Z_ground[x_idx, y_idx] = ground_z_vals[i]

# Get the list of non-NaN indices in Z_ground
non_nan_indices = np.argwhere(~np.isnan(Z_ground))

# Calculate the combined step and slope risk grids
step_risk_grid, slope_risk_grid = calculate_combined_risks(
    Z_ground,
    non_nan_indices,
    max_height_diff=0.04,
    max_slope_degrees=30.0,
    radius=0.5
)
combined_mask = np.isnan(step_risk_grid) & np.isnan(slope_risk_grid)
masked_step_risk = np.ma.masked_array(step_risk_grid, mask=combined_mask)*2.0
masked_slope_risk = np.ma.masked_array(slope_risk_grid, mask=combined_mask)*2.0

# Calculate the sum for non-NaN elements
sum_grid = np.ma.filled(masked_step_risk, 0) + np.ma.filled(masked_slope_risk, 0)
# Create a mask: if both step and slope risks are NaN, then the result should be NaN
both_nan_mask = np.isnan(step_risk_grid) & np.isnan(slope_risk_grid)
# Use np.where to set cells where both risks are NaN to NaN, otherwise use the computed sum
total_risk_grid = np.where(both_nan_mask, np.nan, sum_grid)

# Add obstacle points to the risk grid
for i in range(len(obstacle_x_vals)):
    x_idx = np.digitize(obstacle_x_vals[i], x_edges) - 1
    y_idx = np.digitize(obstacle_y_vals[i], y_edges) - 1
    if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):

        total_risk_grid[x_idx, y_idx] +=2.0  # Mark obstacles as high risk
# Compute the maximum risk value, ignoring NaNs
max_risk = np.nanmax(total_risk_grid)
threshold = 0.4 * max_risk
mask = total_risk_grid > threshold
total_risk_grid[mask] = np.exp(total_risk_grid[mask])

# Interpolate missing (NaN) values in the risk grid
interpolation_radius = 1.5  # Set the interpolation radius
total_risk_grid = interpolate_in_radius(total_risk_grid, interpolation_radius)

# Mask NaN values in total_risk_grid for transparency
masked_total_risk_grid = ma.masked_invalid(total_risk_grid)

# Calculate CVaR for each grid cell
cvar_combined_risk = compute_cvar_cellwise(masked_total_risk_grid, alpha=0.9)
# Define the colormap
colors = [(0.5, 0.5, 0.5), (1, 1, 0), (1, 0, 0)]
cmap = LinearSegmentedColormap.from_list("gray_yellow_red", colors)

# Create a figure and axes object
fig, ax = plt.subplots()

# Plot the risk visualization
c = ax.pcolormesh(X, Y, cvar_combined_risk.T, shading='auto', cmap=cmap, alpha=0.7)
colorbar = fig.colorbar(c, ax=ax, label='Risk Value (0=zero risk, 1= risky)')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Risk Visualization')

plt.show()
