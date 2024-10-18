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
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# Define the number of bins along X and Y axes (the resolution of your grid)
num_bins_x = 50  # Adjust this value to change grid resolution along X-axis
num_bins_y = 50  # Adjust this value to change grid resolution along Y-axis

# Bin the points into grid cells based on X, Y coordinates
statistic, x_edges, y_edges, binnumber = binned_statistic_2d(x, y, z, statistic='mean', bins=[num_bins_x, num_bins_y])

# Create a meshgrid for visualization
x_mid = (x_edges[:-1] + x_edges[1:]) / 2
y_mid = (y_edges[:-1] + y_edges[1:]) / 2
X, Y = np.meshgrid(x_mid, y_mid)

# Plot the 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, statistic.T, cmap='viridis')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Binned 2.5D Elevation Map')

# Display the plot
plt.show()
