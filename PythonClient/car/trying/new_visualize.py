import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

# Load the PLY point cloud file
point_cloud = o3d.io.read_point_cloud('grid_point_cloud.ply')

# Convert the point cloud to a numpy array
points = np.asarray(point_cloud.points)

# Extract X, Y, Z coordinates
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# Define the number of bins (grid resolution)
num_bins = 100

# Compute the 2D histogram to bin the points and calculate the mean of Z for each bin
stat, x_edges, y_edges, bin_numbers = binned_statistic_2d(x, y, z, statistic='median', bins=num_bins)

# Create a meshgrid for visualization
x_mid = (x_edges[:-1] + x_edges[1:]) / 2
y_mid = (y_edges[:-1] + y_edges[1:]) / 2
X, Y = np.meshgrid(x_mid, y_mid)

# Plot the gridded elevation map
plt.figure(figsize=(10, 8))
plt.pcolormesh(X, Y, stat.T, cmap='terrain')
plt.colorbar(label='Average Elevation (Z)')
plt.title("Binned 2.5D Elevation Map from Point Cloud (Averaged per Grid)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()