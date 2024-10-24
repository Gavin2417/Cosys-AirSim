import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy.ma as ma  # For masked arrays

# Load the PLY point cloud files
ground_point_cloud = o3d.io.read_point_cloud('ground_point_cloud.ply')
obstacle_point_cloud = o3d.io.read_point_cloud('obstacle_point_cloud.ply')

# Convert the point clouds to numpy arrays
ground_points = np.asarray(ground_point_cloud.points)
obstacle_points = np.asarray(obstacle_point_cloud.points)

# Extract X, Y, Z coordinates for ground and obstacle points
ground_x_vals = ground_points[:, 0]
ground_y_vals = ground_points[:, 1]
ground_z_vals = ground_points[:, 2]

obstacle_x_vals = obstacle_points[:, 0]
obstacle_y_vals = obstacle_points[:, 1]
obstacle_z_vals = obstacle_points[:, 2]

# Define the grid resolution
grid_resolution = 0.1  # This should match the resolution you want for the grid

# Create grid edges for X and Y based on the range of the ground point cloud values
x_edges = np.arange(min(ground_x_vals), max(ground_x_vals) + grid_resolution, grid_resolution)
y_edges = np.arange(min(ground_y_vals), max(ground_y_vals) + grid_resolution, grid_resolution)

# Create meshgrid for X and Y (for the ground grid)
x_mid = (x_edges[:-1] + x_edges[1:]) / 2  # Midpoints of X bins
y_mid = (y_edges[:-1] + y_edges[1:]) / 2  # Midpoints of Y bins
X, Y = np.meshgrid(x_mid, y_mid)

# Initialize a grid for marking obstacle and ground points with NaN values initially
risk_grid = np.full((len(x_mid), len(y_mid)), np.nan)  # Start with NaN

# Mark grid cells that contain obstacles as 1, leave NaN where no points exist
for i in range(len(obstacle_x_vals)):
    x_idx = np.digitize(obstacle_x_vals[i], x_edges) - 1  # Find the bin index for the x value
    y_idx = np.digitize(obstacle_y_vals[i], y_edges) - 1  # Find the bin index for the y value

    if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
        risk_grid[x_idx, y_idx] = 1.0  # Mark obstacle cells as 1

# Mark grid cells with ground points as 0
for i in range(len(ground_x_vals)):
    x_idx = np.digitize(ground_x_vals[i], x_edges) - 1  # Find the bin index for the x value
    y_idx = np.digitize(ground_y_vals[i], y_edges) - 1  # Find the bin index for the y value

    if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
        if np.isnan(risk_grid[x_idx, y_idx]):  # Only mark as 0 if not already an obstacle (NaN)
            risk_grid[x_idx, y_idx] = 0.0  # Mark ground cells as 0

# Mask NaN values to keep them uncolored
masked_risk_grid = ma.masked_invalid(risk_grid)

# Create a custom colormap from gray (0) to yellow (0.5) back to red (1)
colors = [(0.5, 0.5, 0.5), (1, 1, 0), (1, 0, 0)]  # Gray → Yellow → Red
cmap = LinearSegmentedColormap.from_list("gray_yellow_red", colors)

# Plot the grid using pcolormesh to show ground and obstacle risks, with NaNs left transparent
fig, ax = plt.subplots()
c = ax.pcolormesh(X, Y, masked_risk_grid.T, shading='auto', cmap=cmap, vmin=0, vmax=1)  # Gray (0) for ground, Yellow (0.5) for mid-range, Red (1) for obstacle

# Add a color bar to indicate the risk value (0 for ground, 1 for obstacle)
fig.colorbar(c, ax=ax, label='Risk Value (0=Ground, 1=Obstacle)')

# Set axis labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Ground and Obstacle Risk Map with Transparent NaN Values')

plt.show()
