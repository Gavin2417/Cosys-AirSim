import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

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

def calculate_slope_risk(points, radius=0.5):
    """
    Calculate the slope risk for each LiDAR point based on its surrounding points.
    
    Args:
    - points: Nx3 numpy array of LiDAR points (X, Y, Z)
    - radius: The radius to consider for the neighborhood of each point
    
    Returns:
    - slope_risk: Nx1 array of slope risks for each point (0 to 1 based on 0 to 45-degree slope)
    """
    nbrs = NearestNeighbors(radius=radius).fit(points[:, :2])  # Fit neighbors based on X, Y
    slope_risk = np.zeros(points.shape[0])
    
    for i, point in enumerate(points):
        # Find neighbors within the given radius
        indices = nbrs.radius_neighbors([point[:2]], return_distance=False)[0]
        
        if len(indices) < 3:
            # If less than 3 neighbors, cannot calculate a slope, assign low risk
            slope_risk[i] = 0
            continue
        
        # Get neighboring points
        neighbors = points[indices]
        
        # Fit a plane to the neighboring points (z = ax + by + c)
        A = np.c_[neighbors[:, 0], neighbors[:, 1], np.ones(neighbors.shape[0])]
        coeffs, _, _, _ = np.linalg.lstsq(A, neighbors[:, 2], rcond=None)
        
        # Coefficients a, b of the plane z = ax + by + c
        a, b, _ = coeffs
        
        # Calculate the slope angle in degrees (arctan of the slope magnitude)
        slope_angle = np.degrees(np.arctan(np.sqrt(a**2 + b**2)))
        
        # Map slope risk to 0-1, where 0 degrees is risk 0 and 45 degrees is risk 1
        max_slope_degrees = 45.0
        risk_value = min(slope_angle / max_slope_degrees, 1.0)  # Limit to 1
        
        slope_risk[i] = risk_value
    
    return slope_risk

# Calculate slope risk
slope_risk = calculate_slope_risk(points_xyz)

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
Z_slope_risk = np.full((len(x_mid), len(y_mid)), np.nan)

# Fill the Z grid with your calculated slope risk values
for i in range(len(x_vals)):
    x_idx = np.digitize(x_vals[i], x_edges) - 1  # Find the bin index for the x value
    y_idx = np.digitize(y_vals[i], y_edges) - 1  # Find the bin index for the y value

    if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
        Z_slope_risk[x_idx, y_idx] = slope_risk[i]

# Plot the grid using pcolormesh with slope risk
fig, ax = plt.subplots()
c = ax.pcolormesh(X, Y, Z_slope_risk.T, shading='auto', cmap='viridis')

# Add a color bar to indicate the slope risk values
fig.colorbar(c, ax=ax, label='Slope Risk (0-1)')

# Set axis labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('2D Grid Cell Plot of Slope Risk from PLY File')

plt.show()
