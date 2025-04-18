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


def compute_cvar_cellwise(risk_grid, alpha=0.2, radius=5.0):
    """
    For each valid cell, collect all risk values within `radius` (in grid cells),
    compute the α‑quantile (VaR) of that local sample, then average all local values
    ≥ VaR to get the empirical CVaR.
    """
    rows, cols = risk_grid.shape
    cvar = np.full_like(risk_grid, np.nan)

    # 1) Build a KD‑tree of all valid‐risk cell coords
    valid_mask = ~np.isnan(risk_grid)
    coords = np.column_stack(np.where(valid_mask))
    values = risk_grid[valid_mask]
    tree = cKDTree(coords)

    # 2) For each valid cell, query its neighborhood
    for idx, (r, c) in enumerate(coords):
        neigh_idx = tree.query_ball_point((r, c), radius)
        local_vals = values[neigh_idx]
        if local_vals.size == 0:
            continue

        # 3) Empirical VaR (α‑quantile)
        var = np.quantile(local_vals, alpha)
        # 4) Average the tail ≥ VaR
        tail = local_vals[local_vals >= var]
        cvar[r, c] = tail.mean() if tail.size else var

    return cvar