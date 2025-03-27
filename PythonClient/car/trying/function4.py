import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
import numpy.ma as ma
from scipy.stats import norm

def calculate_combined_risks(Z_grid, non_nan_indices, max_height_diff=0.4, max_slope_degrees=30.0, radius=0.3):
    """
    Calculate step and slope risks for a grid based on height differences and gradients.
    """
    # Initialize grids for step and slope risks
    step_risk_grid = np.full_like(Z_grid, np.nan)
    slope_risk_grid = np.full_like(Z_grid, np.nan)

    # Define offsets for 8 neighbors
    neighbors = np.array([
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1), (1, 0), (1, 1)
    ])

    # Precompute max slope in radians
    max_slope_radians = np.deg2rad(max_slope_degrees)

    for x, y in non_nan_indices:
        z_center = Z_grid[x, y]
        if np.isnan(z_center):
            continue

        # Calculate neighbors' indices and filter valid ones
        neighbor_indices = neighbors + np.array([x, y])
        valid_indices = neighbor_indices[
            (0 <= neighbor_indices[:, 0]) & (neighbor_indices[:, 0] < Z_grid.shape[0]) &
            (0 <= neighbor_indices[:, 1]) & (neighbor_indices[:, 1] < Z_grid.shape[1])
        ]

        # Extract neighbor heights
        neighbor_heights = [
            Z_grid[nx, ny]
            for nx, ny in valid_indices
            if not np.isnan(Z_grid[nx, ny])
        ]

        if neighbor_heights:
            # Calculate step risk
            max_diff = np.max(np.abs(np.array(neighbor_heights) - z_center))
            step_risk = min(max_diff / max_height_diff, 1.0)
            step_risk_grid[x, y] = step_risk

            # Calculate slope risk
            xy_distance = np.sqrt((1 * radius) ** 2 + (1 * radius) ** 2)
            slopes = [
                np.arctan2(abs(z_neighbor - z_center), xy_distance)
                for z_neighbor in neighbor_heights
            ]
            max_slope = np.max(slopes)
            slope_risk = min(max_slope / max_slope_radians, 1.0)
            slope_risk_grid[x, y] = slope_risk
        else:
            step_risk_grid[x, y] = 0
            slope_risk_grid[x, y] = 0

    return step_risk_grid, slope_risk_grid


def compute_cvar_cellwise(risk_grid, alpha=0.20):
    """
    Optimized computation of CVaR for a grid, assuming a normal distribution for risks.
    """
    # Mask NaN values to avoid unnecessary computation
    valid_mask = ~np.isnan(risk_grid)
    
    mu = np.zeros_like(risk_grid)
    mu[valid_mask] = risk_grid[valid_mask]
    
    sigma = 0.1  # Adjust standard deviation as needed
    phi = norm.pdf(norm.ppf(alpha))
    cvar_grid = np.full_like(risk_grid, np.nan)  # Start with a grid of NaNs

    # Compute CVaR only for valid cells
    cvar_grid[valid_mask] = np.clip(mu[valid_mask] + sigma * (phi / (1 - alpha)),0, 40)
    
    return cvar_grid