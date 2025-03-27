import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
import numpy.ma as ma
from scipy.stats import norm

def calculate_combined_risks(Z_grid, non_nan_indices, max_height_diff=0.5, max_slope_degrees=30.0,
                             radius=0.3, lambda_step=0.02, gamma_slope=0.3):
    """
    Calculate step and slope risks for a grid based on height differences and gradients,
    applying an exponential transformation so that risk values are unbounded.
    
    For step risk:
      - Compute the maximum absolute height difference between the center cell and its neighbors.
      - Normalize it by max_height_diff.
      - Transform it as: step_risk = exp(lambda_step * normalized_difference) - 1.
      
    For slope risk:
      - Compute the maximum slope (using arctan2 with a fixed xy distance).
      - Normalize by the safe maximum slope in radians.
      - Transform it as: slope_risk = exp(gamma_slope * normalized_slope) - 1.
      
    Parameters:
        Z_grid (np.ndarray): The grid of height values.
        non_nan_indices (array-like): Indices (x, y) in the grid where Z_grid is not NaN.
        max_height_diff (float): The reference height difference (safe step threshold).
        max_slope_degrees (float): The safe slope limit in degrees.
        radius (float): Distance used for computing the slope.
        lambda_step (float): Scaling factor for the exponential step risk transformation.
        gamma_slope (float): Scaling factor for the exponential slope risk transformation.
    
    Returns:
        (np.ndarray, np.ndarray): Two grids (step risk, slope risk) with transformed risk values.
    """
    # Initialize grids for step and slope risks
    step_risk_grid = np.full_like(Z_grid, np.nan)
    slope_risk_grid = np.full_like(Z_grid, np.nan)

    # Define offsets for 8 neighbors
    neighbors = np.array([
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ])

    # Convert safe slope limit to radians.
    max_slope_radians = np.deg2rad(max_slope_degrees)

    for x, y in non_nan_indices:
        z_center = Z_grid[x, y]
        if np.isnan(z_center):
            continue

        # Calculate neighbor indices and filter valid ones
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
            # Calculate step risk.
            max_diff = np.max(np.abs(np.array(neighbor_heights) - z_center))
            # Normalize the difference (values can exceed 1 if max_diff > max_height_diff)
            step_norm = max_diff / max_height_diff
            # Exponential transformation: unbounded risk measure
            step_risk = np.exp(lambda_step * step_norm) - 1
            step_risk_grid[x, y] = step_risk

            # Calculate slope risk.
            # Here, we use a fixed distance in the xy-plane for slope computation.
            xy_distance = np.sqrt(radius**2 + radius**2)
            slopes = [
                np.arctan2(abs(z_neighbor - z_center), xy_distance)
                for z_neighbor in neighbor_heights
            ]
            max_slope = np.max(slopes)
            slope_norm = max_slope / max_slope_radians
            # Exponential transformation for slope risk.
            slope_risk = np.exp(gamma_slope * slope_norm) - 1
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
    cvar_grid[valid_mask] = np.clip(mu[valid_mask] + sigma * (phi / (1 - alpha)), 0, 1)

    return cvar_grid