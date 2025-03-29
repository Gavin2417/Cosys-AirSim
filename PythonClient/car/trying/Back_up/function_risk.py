import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
import numpy.ma as ma
from scipy.stats import norm
import torch

def calculate_combined_risks(Z_grid, non_nan_indices, max_height_diff=0.5, max_slope_degrees=30.0, radius=0.3):
    """
    Calculate step and slope risks on GPU for enhanced performance.
    """
    # Transfer grid to GPU
    Z_grid_gpu = torch.tensor(Z_grid, device='cuda', dtype=torch.float32)
    step_risk_grid = torch.full_like(Z_grid_gpu, float('nan'))
    slope_risk_grid = torch.full_like(Z_grid_gpu, float('nan'))
    
    neighbors = torch.tensor([
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),         (0, 1),
        (1, -1), (1, 0), (1, 1)
    ], device='cuda')
    
    max_slope_radians = torch.deg2rad(torch.tensor(max_slope_degrees, device='cuda'))

    for x, y in non_nan_indices:
        z_center = Z_grid_gpu[x, y]
        if torch.isnan(z_center):
            continue

        # Calculate neighbors
        neighbor_indices = neighbors + torch.tensor([x, y], device='cuda')
        valid_mask = (
            (0 <= neighbor_indices[:, 0]) & (neighbor_indices[:, 0] < Z_grid_gpu.shape[0]) &
            (0 <= neighbor_indices[:, 1]) & (neighbor_indices[:, 1] < Z_grid_gpu.shape[1])
        )
        valid_indices = neighbor_indices[valid_mask]
        
        neighbor_heights = Z_grid_gpu[valid_indices[:, 0], valid_indices[:, 1]]
        valid_heights = neighbor_heights[~torch.isnan(neighbor_heights)]
        
        if valid_heights.numel() > 0:
            max_diff = torch.max(torch.abs(valid_heights - z_center))
            step_risk_grid[x, y] = min(max_diff / max_height_diff, 1.0)
            
            xy_distance = torch.sqrt(torch.tensor(2 * radius**2))
            slopes = torch.arctan2(torch.abs(valid_heights - z_center), xy_distance)
            max_slope = torch.max(slopes)
            slope_risk_grid[x, y] = min(max_slope / max_slope_radians, 1.0)
        else:
            step_risk_grid[x, y] = 0
            slope_risk_grid[x, y] = 0

    return step_risk_grid.cpu().numpy(), slope_risk_grid.cpu().numpy()



def compute_cvar_cellwise(risk_grid, alpha=0.75):
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