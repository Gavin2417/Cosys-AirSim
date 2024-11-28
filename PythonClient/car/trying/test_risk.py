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
# Load the PLY point cloud files
ground_point_cloud = o3d.io.read_point_cloud('ground_point_cloud.ply')
obstacle_point_cloud = o3d.io.read_point_cloud('obstacle_point_cloud.ply')

# Convert the point clouds to numpy arrays
ground_points = np.asarray(ground_point_cloud.points)
obstacle_points = np.asarray(obstacle_point_cloud.points)

# Extract X, Y, Z for ground points
ground_x_vals = ground_points[:, 0]
ground_y_vals = ground_points[:, 1]
ground_z_vals = ground_points[:, 2]

obstacle_x_vals = obstacle_points[:, 0]
obstacle_y_vals = obstacle_points[:, 1]
obstacle_z_vals = obstacle_points[:, 2]

# Define the grid resolution
grid_resolution = 0.1

# Create grid edges for X and Y based on the range of ground points
x_edges = np.arange(min(ground_x_vals), max(ground_x_vals) + grid_resolution, grid_resolution)
y_edges = np.arange(min(ground_y_vals), max(ground_y_vals) + grid_resolution, grid_resolution)

# Create meshgrid for X and Y (for ground)
x_mid = (x_edges[:-1] + x_edges[1:]) / 2  # Midpoints of X bins
y_mid = (y_edges[:-1] + y_edges[1:]) / 2  # Midpoints of Y bins
X, Y = np.meshgrid(x_mid, y_mid)

# Initialize an empty Z grid for ground points
Z_ground = np.full((len(x_mid), len(y_mid)), np.nan)

# Fill the Z grid for ground points
for i in range(len(ground_x_vals)):
    x_idx = np.digitize(ground_x_vals[i], x_edges) - 1
    y_idx = np.digitize(ground_y_vals[i], y_edges) - 1
    if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
        Z_ground[x_idx, y_idx] = ground_z_vals[i]

# Get the list of non-NaN indices in Z_ground
non_nan_indices = np.argwhere(~np.isnan(Z_ground))

# Calculate the combined step and slope risk grids
step_risk_grid, slope_risk_grid = calculate_combined_risks(Z_ground, non_nan_indices, max_height_diff=0.05, max_slope_degrees=30.0, radius=0.5)

combined_mask = np.isnan(step_risk_grid) & np.isnan(slope_risk_grid)
masked_step_risk = np.ma.masked_array(step_risk_grid, mask=combined_mask)
masked_slope_risk = np.ma.masked_array(slope_risk_grid, mask=combined_mask)

# Calculate the mean for non-NaN elements
total_risk_grid = np.ma.mean([masked_step_risk, masked_slope_risk], axis=0).filled(np.nan)

# Add obstacle points to the risk grid
for i in range(len(obstacle_x_vals)):
    x_idx = np.digitize(obstacle_x_vals[i], x_edges) - 1
    y_idx = np.digitize(obstacle_y_vals[i], y_edges) - 1
    if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
        total_risk_grid[x_idx, y_idx] = 1.0  # Mark obstacles as high risk

# Mask NaN values in total_risk_grid for transparency
masked_total_risk_grid = ma.masked_invalid(total_risk_grid)

# Calculate CVaR for each grid cell
cvar_combined_risk = compute_cvar_cellwise(masked_total_risk_grid)
print('cvar_combined_risK finished')
start_point = np.array([-5, 0])

destination_point = np.array([15,-10])

# Create a custom colormap from gray to yellow to red
colors = [(0.5, 0.5, 0.5), (1, 1, 0), (1, 0, 0)]  # Gray → Yellow → Red
cmap = LinearSegmentedColormap.from_list("gray_yellow_red", colors)

# # Plot the total risk grid
# fig, ax = plt.subplots()
# c = ax.pcolormesh(X, Y, cvar_combined_risk.T, shading='auto', cmap=cmap, vmin=0, vmax=1)
# fig.colorbar(c, ax=ax, label='Risk Value (0=Ground, 1=Obstacle)')

# # Set axis labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Total Risk Grid with Ground and Obstacle Data')

plt.show()
from heapq import heappop, heappush

def heuristic(a, b):
    """Calculate the Manhattan distance as the heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Convert start and destination points to grid indices
start_idx = (
    np.digitize(start_point[0], x_edges) - 1,
    np.digitize(start_point[1], y_edges) - 1,
)
goal_idx = (
    np.digitize(destination_point[0], x_edges) - 1,
    np.digitize(destination_point[1], y_edges) - 1,
)
def astar_fixed(cost_map, start, goal):
    """Perform A* algorithm, avoiding impassable NaN regions."""
    rows, cols = cost_map.shape
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heappop(open_set)

        if current == goal:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return reversed path

        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4 neighbors
            neighbor = (x + dx, y + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # Ignore cells with NaN or infinite costs
                if np.isnan(cost_map[neighbor[0], neighbor[1]]):
                    continue

                tentative_g_score = g_score[current] + cost_map[neighbor[0], neighbor[1]]
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

    return []  # Return empty list if no path is found

# Run the updated A* algorithm
path = astar_fixed(cvar_combined_risk, start_idx, goal_idx)

# Visualize the path on the updated risk map
fig, ax = plt.subplots()
c = ax.pcolormesh(X, Y, cvar_combined_risk.T, shading='auto', cmap=cmap, vmin=0, vmax=1)
fig.colorbar(c, ax=ax, label='Risk Value (0=Ground, 1=Obstacle)')

# Overlay the fixed path
if path:
    path_x = [x_mid[p[0]] for p in path]
    path_y = [y_mid[p[1]] for p in path]
    ax.plot(path_x, path_y, color='blue', linewidth=2, label='A* Path (Fixed)')
    ax.legend()

# Set axis labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Total Risk Grid with Fixed A* Path')

plt.show()