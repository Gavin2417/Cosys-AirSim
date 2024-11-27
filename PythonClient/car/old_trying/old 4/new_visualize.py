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

# Visualize the slope risk
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

# Plot the points, using slope risk as the color map
scatter = ax.scatter(x_vals, y_vals, z_vals, c=slope_risk, cmap='viridis')

# Add color bar
plt.colorbar(scatter, ax=ax, label='Slope Risk (0-1)')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Slope Risk of LiDAR Points')

plt.show()
