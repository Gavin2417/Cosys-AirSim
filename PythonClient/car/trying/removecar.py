import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import binned_statistic_2d

# Load the PLY point cloud file
point_cloud = o3d.io.read_point_cloud('grid_point_cloud.ply')

# Convert the point cloud to a numpy array
points = np.asarray(point_cloud.points)

# Extract X, Y, Z coordinates
x_vals = points[:, 0]
y_vals = points[:, 1]
z_vals = points[:, 2]

# Define the grid resolution (must match the resolution you used to create the grid)
grid_resolution = 0.1  # This should be the same as the resolution you used

# Create grid edges for X and Y based on the range of your X and Y values
x_edges = np.arange(min(x_vals), max(x_vals) + grid_resolution, grid_resolution)
y_edges = np.arange(min(y_vals), max(y_vals) + grid_resolution, grid_resolution)

# Create meshgrid for X and Y
x_mid = (x_edges[:-1] + x_edges[1:]) / 2  # Midpoints of X bins
y_mid = (y_edges[:-1] + y_edges[1:]) / 2  # Midpoints of Y bins
X, Y = np.meshgrid(x_mid, y_mid)

# Initialize an empty Z grid
Z = np.full((len(x_mid), len(y_mid)), np.nan)

# Fill the Z grid with your pre-calculated Z values
for i in range(len(x_vals)):
    x_idx = np.digitize(x_vals[i], x_edges) - 1  # Find the bin index for the x value
    y_idx = np.digitize(y_vals[i], y_edges) - 1  # Find the bin index for the y value

    if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
        Z[x_idx, y_idx] = z_vals[i]

# Plot the grid using pcolormesh
fig, ax = plt.subplots()
c = ax.pcolormesh(X, Y, Z.T, shading='auto', cmap='terrain')

# Add a color bar to indicate the Z values (heights)
fig.colorbar(c, ax=ax, label='Average Z Value (Height)')

# Set axis labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('2D elevation map')

plt.show()
