import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
import numpy as np
from scipy.spatial import cKDTree
import numpy.ma as ma  
from scipy.stats import norm  
def calculate_combined_risks(Z_grid, non_nan_indices, max_height_diff=0.5, max_slope_degrees=30.0, radius=0.3):
    """
    Calculate both step and slope risks based on height differences and gradients between neighboring cells,
    considering only neighbors within a specified radius for slope risk.
    """
    # Initialize grids for step and slope risks
    step_risk_grid = np.full_like(Z_grid, np.nan)
    slope_risk_grid = np.full_like(Z_grid, np.nan)

    # Define offsets for 8 neighbors for step risk
    all_neighbors_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    # Convert max slope degrees to radians for slope normalization
    max_slope_radians = np.deg2rad(max_slope_degrees)

    # Filter offsets for slope risk to include only those within the radius
    filtered_slope_offsets = [
        (dx, dy) for dx, dy in all_neighbors_offsets
        if np.sqrt((dx * radius) ** 2 + (dy * radius) ** 2) <= radius
    ]

    for x, y in non_nan_indices:
        z_center = Z_grid[x, y]
        if np.isnan(z_center):
            continue

        # Initialize lists to store height differences and slopes
        height_diffs = []
        slopes = []

        # Calculate height differences with 8 neighbors for step risk
        for dx, dy in all_neighbors_offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < Z_grid.shape[0] and 0 <= ny < Z_grid.shape[1]:
                z_neighbor = Z_grid[nx, ny]
                if not np.isnan(z_neighbor):
                    height_diffs.append(abs(z_neighbor - z_center))

        # Calculate slopes with filtered neighbors for slope risk
        for dx, dy in filtered_slope_offsets:
            
            nx, ny = x + dx, y + dy
            if 0 <= nx < Z_grid.shape[0] and 0 <= ny < Z_grid.shape[1]:
                z_neighbor = Z_grid[nx, ny]
                if not np.isnan(z_neighbor):
                    # Calculate XY distance
                    xy_distance = np.sqrt((dx * radius) ** 2 + (dy * radius) ** 2)
                    # Calculate the slope angle
                    slope_angle = np.arctan2(abs(z_neighbor - z_center), xy_distance)
                    slopes.append(slope_angle)

        # Calculate step risk by normalizing max height difference
        if height_diffs:
            max_diff = np.max(height_diffs)
            step_risk = min(max_diff / max_height_diff, 1.0)
            step_risk_grid[x, y] = step_risk
        else:
            step_risk_grid[x, y] = 0

        # Calculate slope risk by normalizing max slope angle
        if slopes:
            max_slope = max(slopes)
            slope_risk = min(max_slope / max_slope_radians, 1.0)
            slope_risk_grid[x, y] = slope_risk
        else:
            slope_risk_grid[x, y] = 0

    return step_risk_grid, slope_risk_grid

def compute_cvar_cellwise(risk_grid, alpha=0.75):
    cvar_grid = np.full_like(risk_grid, np.nan)
    
    for i in range(risk_grid.shape[0]):
        for j in range(risk_grid.shape[1]):
            # Skip NaN cells
            if np.isnan(risk_grid[i, j]):
                continue

            # Assuming a normal distribution for each cell's risk
            mu = risk_grid[i, j]  # Mean risk
            sigma = 0.1  # Small standard deviation; adjust as needed

            # Calculate the CVaR for the cell
            phi = norm.pdf(norm.ppf(alpha))
            cvar = mu + sigma * (phi / (1 - alpha))
            cvar_grid[i, j] = np.clip(cvar, 0, 1)  # Ensure CVaR is in [0, 1]

    return cvar_grid