import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm
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

# Recreate the points array to be Nx3
points_xyz = np.vstack((ground_x_vals, ground_y_vals, ground_z_vals)).T

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
obstacle_grid = np.full((len(x_mid), len(y_mid)), np.nan)  # Start with NaN

# Mark grid cells that contain obstacles as 1, leave NaN where no points exist
for i in range(len(obstacle_x_vals)):
    x_idx = np.digitize(obstacle_x_vals[i], x_edges) - 1  # Find the bin index for the x value
    y_idx = np.digitize(obstacle_y_vals[i], y_edges) - 1  # Find the bin index for the y value

    if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
        obstacle_grid[x_idx, y_idx] = 1.0  # Mark obstacle cells as 1

# Mark grid cells with ground points as 0
for i in range(len(ground_x_vals)):
    x_idx = np.digitize(ground_x_vals[i], x_edges) - 1  # Find the bin index for the x value
    y_idx = np.digitize(ground_y_vals[i], y_edges) - 1  # Find the bin index for the y value

    if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
        if np.isnan(obstacle_grid[x_idx, y_idx]):  # Only mark as 0 if not already an obstacle (NaN)
            obstacle_grid[x_idx, y_idx] = 0.0  # Mark ground cells as 0

# Mask NaN values to keep them uncolored
masked_obstacle_grid = ma.masked_invalid(obstacle_grid)

# Function to calculate slope risk
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

# Function to calculate step risk
def calculate_step_risk(points, grid_resolution=0.1, max_height_diff=0.5, ground_threshold=-0.55):
    """
    Calculate step risk based on height difference between adjacent cells in the grid map.
    Also considers points that are significantly lower than a baseline ground level.
    
    Args:
    - points: Nx3 numpy array of LiDAR points (X, Y, Z)
    - grid_resolution: The grid resolution used for binning points.
    - max_height_diff: The maximum height difference to be considered for risk (e.g., 0.5 meters).
    - ground_threshold: The threshold for ground detection (e.g., -0.55 meters).
    
    Returns:
    - step_risk: Nx1 array of step risks for each point (0 to 1 based on height differences).
    """
    
    # Create grid edges for X and Y based on the range of your X and Y values
    x_edges = np.arange(min(ground_x_vals), max(ground_x_vals) + grid_resolution, grid_resolution)
    y_edges = np.arange(min(ground_y_vals), max(ground_y_vals) + grid_resolution, grid_resolution)

    # Create meshgrid for X and Y
    x_mid = (x_edges[:-1] + x_edges[1:]) / 2  # Midpoints of X bins
    y_mid = (y_edges[:-1] + y_edges[1:]) / 2  # Midpoints of Y bins
    X, Y = np.meshgrid(x_mid, y_mid)

    # Initialize an empty Z grid
    Z_grid = np.full((len(x_mid), len(y_mid)), np.nan)

    # Fill the Z grid with the height values
    for i in range(len(ground_x_vals)):
        x_idx = np.digitize(ground_x_vals[i], x_edges) - 1  # Find the bin index for the x value
        y_idx = np.digitize(ground_y_vals[i], y_edges) - 1  # Find the bin index for the y value

        if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
            Z_grid[x_idx, y_idx] = ground_z_vals[i]

    # Initialize a step risk grid
    step_risk_grid = np.full_like(Z_grid, np.nan)

    # Calculate step risk based on height differences between neighboring cells
    for x in range(1, len(x_mid) - 1):
        for y in range(1, len(y_mid) - 1):
            # Get the current cell height
            z_center = Z_grid[x, y]
            if np.isnan(z_center):
                continue

            # Get neighboring heights (north, south, west, east)
            neighbors = [
                Z_grid[x-1, y], Z_grid[x+1, y], Z_grid[x, y-1], Z_grid[x, y+1]
            ]
            neighbors = [z for z in neighbors if not np.isnan(z)]  # Ignore NaNs

            if neighbors:
                # Calculate the height difference with the neighbors
                max_diff = np.max(np.abs(np.array(neighbors) - z_center))

                # Normalize the risk based on the max height difference
                step_risk = min(max_diff / max_height_diff, 1.0)  # Scale to 0-1
            else:
                step_risk = 0  # If no neighbors, assume no risk

            # Additional risk: Check if the height is significantly below the ground threshold
            if z_center > ground_threshold:
                # Height difference with the baseline ground
                ground_risk = min(np.abs(z_center - ground_threshold) / max_height_diff, 1.0)
                # Combine the step risk and the ground risk
                step_risk = max(step_risk, ground_risk)
                
            step_risk_grid[x, y] = step_risk

    return step_risk_grid, X, Y

# Calculate slope risk
slope_risk = calculate_slope_risk(points_xyz)

# Define the grid resolution (must match the resolution you used to create the grid)
grid_resolution = 0.1

# Create grid edges for X and Y based on the range of your X and Y values
x_edges = np.arange(min(ground_x_vals), max(ground_x_vals) + grid_resolution, grid_resolution)
y_edges = np.arange(min(ground_y_vals), max(ground_y_vals) + grid_resolution, grid_resolution)

# Create meshgrid for X and Y
x_mid = (x_edges[:-1] + x_edges[1:]) / 2  # Midpoints of X bins
y_mid = (y_edges[:-1] + y_edges[1:]) / 2  # Midpoints of Y bins
X, Y = np.meshgrid(x_mid, y_mid)

# Initialize an empty Z grid for slope risk
Z_slope_risk = np.full((len(x_mid), len(y_mid)), np.nan)

# Fill the Z grid with your calculated slope risk values
for i in range(len(ground_x_vals)):
    x_idx = np.digitize(ground_x_vals[i], x_edges) - 1  # Find the bin index for the x value
    y_idx = np.digitize(ground_y_vals[i], y_edges) - 1  # Find the bin index for the y value

    if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
        Z_slope_risk[x_idx, y_idx] = slope_risk[i]

# Calculate step risk
step_risk_grid, X, Y = calculate_step_risk(points_xyz, grid_resolution=0.1, max_height_diff=0.5, ground_threshold=-0.55)

# Combine slope risk, step risk, and obstacle grid
# Compute the average risk (excluding NaN values)
combined_risk_grid = np.nanmean(np.array([Z_slope_risk, step_risk_grid, obstacle_grid]), axis=0)

# Apply CVaR locally on each grid cell
def compute_cvar_cellwise(risk_grid, alpha=0.75):
    """
    Compute the CVaR for each grid cell independently.

    Args:
    - risk_grid: 2D array of combined risks (slope + step + obstacles).
    - alpha: Confidence level for CVaR calculation (e.g., 0.75 for 75%).

    Returns:
    - cvar_grid: 2D array of CVaR-adjusted risks.
    """
    cvar_grid = np.full_like(risk_grid, np.nan)
    
    for i in range(risk_grid.shape[0]):
        for j in range(risk_grid.shape[1]):
            # Skip NaN cells
            if np.isnan(risk_grid[i, j]):
                continue

            # For each cell, assume a normal distribution with:
            mu = risk_grid[i, j]  # Mean is the cell's risk
            sigma = 0.1  # Standard deviation assumed small; you can adjust this

            # Compute PDF and CDF for the standard normal distribution
            phi = norm.pdf(norm.ppf(alpha))
            cvar = mu + sigma * (phi / (1 - alpha))
            cvar_grid[i, j] = np.clip(cvar, 0, 1)  # Ensure the CVaR is in the [0, 1] range

    return cvar_grid

# Calculate CVaR for each grid cell
cvar_combined_risk = compute_cvar_cellwise(combined_risk_grid)

# Create a custom colormap from gray to yellow to red
colors = [(0.5, 0.5, 0.5), (1, 1, 0), (1, 0, 0)]  # Gray → Yellow → Red
cmap = LinearSegmentedColormap.from_list("gray_yellow_red", colors)

# Plot the combined risk with CVaR applied
fig, ax = plt.subplots(figsize=(7, 6))

c = ax.pcolormesh(X, Y, cvar_combined_risk.T, shading='auto', cmap=cmap, vmin=0, vmax=1)
fig.colorbar(c, ax=ax, label='CVaR Combined Risk (0-1)', ticks=[0, 0.25, 0.5, 0.75, 1])

# Set axis labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('CVaR')

plt.show()
