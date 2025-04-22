import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
import numpy.ma as ma
from scipy.stats import norm

def calculate_combined_risks(Z_grid, non_nan_indices, max_height_diff=0.4, max_slope_degrees=30.0, radius=0.3):
    """
    Vectorized step & slope risk over an 8‐neighbor window.
    The non_nan_indices argument is kept for compatibility but not used.
    """
    # Precompute constants
    max_slope_rad = np.deg2rad(max_slope_degrees)
    diag_dist = np.sqrt(radius**2 + radius**2)

    # Define the 8 neighbor shifts
    shifts = [
        ( 0,  1), ( 0, -1),
        ( 1,  0), (-1,  0),
        ( 1,  1), ( 1, -1),
        (-1,  1), (-1, -1),
    ]

    # Compute absolute height differences for each shift
    diffs = []
    for dx, dy in shifts:
        shifted = np.roll(np.roll(Z_grid, dx, axis=0), dy, axis=1)
        diffs.append(np.abs(shifted - Z_grid))

    all_diffs = np.stack(diffs, axis=0)

    # Mask out differences where either cell was NaN
    nan_mask = np.isnan(Z_grid)
    all_diffs[:, nan_mask] = 0

    # Maximum neighbor difference per cell
    max_diff = np.max(all_diffs, axis=0)

    # Step risk: normalized and capped
    step_risk = np.minimum(max_diff / max_height_diff, 1.0)

    # Slope risk: arctan of gradient over diagonal distance, normalized and capped
    slope_risk = np.minimum((np.arctan(max_diff / diag_dist) / max_slope_rad), 1.0)

    # Restore NaNs where input was NaN
    step_risk[nan_mask] = np.nan
    slope_risk[nan_mask] = np.nan

    return step_risk, slope_risk


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

        clean = np.ma.compressed(local_vals)    # gives a 1‑D ndarray of just the unmasked values
        if clean.size == 0:
            continue
        var = np.quantile(clean, alpha)
        # 4) Average the tail ≥ VaR
        tail = local_vals[local_vals >= var]
        cvar[r, c] = tail.mean() if tail.size else var

    return cvar