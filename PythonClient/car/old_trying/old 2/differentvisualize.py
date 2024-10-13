
# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata

# # Load the PLY point cloud file
# point_cloud = o3d.io.read_point_cloud('grid_point_cloud.ply')

# # Convert the point cloud to a numpy array
# points = np.asarray(point_cloud.points)

# # Extract X, Y, Z coordinates
# x = points[:, 0]
# y = points[:, 1]
# z = points[:, 2]

# # Define grid resolution (number of bins in x and y directions)
# grid_x, grid_y = np.mgrid[min(x):max(x):200j, min(y):max(y):200j]

# # Interpolate the Z values (elevation) over the grid using griddata
# grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

# # Plot the gridded elevation map
# plt.figure(figsize=(10, 8))
# plt.imshow(grid_z, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='terrain')
# plt.colorbar(label='Elevation (Z)')
# plt.title("Gridded 2.5D Elevation Map from Point Cloud")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()



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

# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import binned_statistic_2d
# from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkit

# # Load the PLY point cloud file
# point_cloud = o3d.io.read_point_cloud('grid_point_cloud.ply')

# # Convert the point cloud to a numpy array
# points = np.asarray(point_cloud.points)

# # Extract X, Y, Z coordinates
# x = points[:, 0]
# y = points[:, 1]
# z = points[:, 2]

# # Define the number of bins (grid resolution)
# num_bins = 100

# # Compute the 2D histogram to bin the points and calculate the mean of Z for each bin
# stat, x_edges, y_edges, bin_numbers = binned_statistic_2d(x, y, z, statistic='median', bins=num_bins)

# # Create a meshgrid for visualization
# x_mid = (x_edges[:-1] + x_edges[1:]) / 2
# y_mid = (y_edges[:-1] + y_edges[1:]) / 2
# X, Y = np.meshgrid(x_mid, y_mid)

# # Create a figure for the 3D plot
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the 3D surface
# surf = ax.plot_surface(X, Y, stat.T, cmap='terrain', edgecolor='none')

# # Add colorbar and labels
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation (Z)')
# ax.set_title("Binned 3D Elevation Map from Point Cloud")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# # Show the plot
# plt.show()

