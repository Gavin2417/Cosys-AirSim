# import setup_path
import os, math, time, heapq
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d, norm
from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure, distance_transform_edt
from scipy.spatial import cKDTree
from matplotlib.colors import LinearSegmentedColormap
import numpy.ma as ma
import cosysairsim as airsim
from linefit import ground_seg
from function4 import calculate_combined_risks, compute_cvar_cellwise
# ---------------------------------------------------------------------------
# Interpolation: Fill in missing (NaN) grid cells using nearby valid cells.
# ---------------------------------------------------------------------------
def interpolate_in_radius(grid, radius):
    """
    Interpolates NaN values in a grid using valid points within a specified radius.
    """
    valid = ~np.isnan(grid)
    valid_coords = np.column_stack(np.where(valid))
    valid_values = grid[valid]
    tree = cKDTree(valid_coords)
    nan_coords = np.column_stack(np.where(np.isnan(grid)))
    for coord in nan_coords:
        neighbors = tree.query_ball_point(coord, radius)
        if neighbors:
            neighbor_coords = valid_coords[neighbors]
            values = valid_values[neighbors]
            distances = np.linalg.norm(neighbor_coords - coord, axis=1) + 1e-6
            weights = 1.0 / distances
            grid[coord[0], coord[1]] = np.sum(weights * values) / np.sum(weights)
    return grid

# ---------------------------------------------------------------------------
# Helper: Filter points within a given radius.
# ---------------------------------------------------------------------------
def filter_points_by_radius(points, center, radius):
    distances = np.linalg.norm(points[:, :2] - center, axis=1)
    return points[distances <= radius]

# ---------------------------------------------------------------------------
# A* Search Helper Functions
# ---------------------------------------------------------------------------
def is_valid(row, col, grid):
    return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]

def is_unblocked(grid, row, col, threshold):
    return (not np.isnan(grid[row, col])) and (grid[row, col] < threshold)

def calculate_h_value(row, col, dest):
    return np.sqrt((row - dest[0]) ** 2 + (col - dest[1]) ** 2)

def trace_path(cell_details, dest):
    path = []
    row, col = dest
    while True:
        path.append((row, col))
        parent = cell_details[row, col]
        if (row, col) == parent:
            break
        row, col = parent
    path.reverse()
    return path

def a_star_search(risk_grid, start_idx, dest_idx):
    rows, cols = risk_grid.shape
    max_risk = np.nanmax(risk_grid)
    threshold = 0.8 * max_risk if not np.isnan(max_risk) else 6.0

    open_list = []
    heapq.heappush(open_list, (0.0, start_idx))
    g_scores = np.full((rows, cols), np.inf)
    g_scores[start_idx] = 0
    f_scores = np.full((rows, cols), np.inf)
    f_scores[start_idx] = calculate_h_value(*start_idx, dest_idx)
    cell_details = np.full((rows, cols), None, dtype=object)
    for i in range(rows):
        for j in range(cols):
            cell_details[i, j] = (i, j)
    
    while open_list:
        _, current = heapq.heappop(open_list)
        if current == dest_idx:
            return trace_path(cell_details, dest_idx)
        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc, risk_grid) and is_unblocked(risk_grid, nr, nc, threshold):
                tentative_g = g_scores[current] + risk_grid[nr, nc]
                if tentative_g < g_scores[nr, nc]:
                    g_scores[nr, nc] = tentative_g
                    f_scores[nr, nc] = tentative_g + calculate_h_value(nr, nc, dest_idx)
                    heapq.heappush(open_list, (f_scores[nr, nc], (nr, nc)))
                    cell_details[nr, nc] = current
    return None

# ---------------------------------------------------------------------------
# Smoothing Function: Smoothens a path using a moving average filter.
# ---------------------------------------------------------------------------
def smooth_path(path, window_size=5):
    """
    Smooths a sequence of (x,y) points using a simple moving average filter.
    """
    path = np.array(path)
    n_points = len(path)
    if n_points < window_size:
        return path
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2
    smoothed = [np.mean(path[max(0, i-half_window):min(n_points, i+half_window+1)], axis=0)
                for i in range(n_points)]
    return np.array(smoothed)

# ---------------------------------------------------------------------------
# Lidar and Vehicle Pose Handling
# ---------------------------------------------------------------------------
class lidarTest:
    def __init__(self, lidar_name, vehicle_name):
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.vehicleName = vehicle_name
        self.lidarName = lidar_name
        self.lastlidarTimeStamp = 0

    def get_data(self, gpulidar):
        if gpulidar:
            lidarData = self.client.getGPULidarData(self.lidarName, self.vehicleName)
        else:
            lidarData = self.client.getLidarData(self.lidarName, self.vehicleName)
        if lidarData.time_stamp != self.lastlidarTimeStamp:
            self.lastlidarTimeStamp = lidarData.time_stamp
            if len(lidarData.point_cloud) < 2:
                return None, None
            points = np.array(lidarData.point_cloud, dtype=np.float32)
            num_dims = 5 if gpulidar else 3
            points = points.reshape((-1, num_dims))
            if not gpulidar:
                points = points * np.array([1, -1, 1])
            return points, lidarData.time_stamp
        return None, None

    def get_vehicle_pose(self):
        vehicle_pose = self.client.simGetVehiclePose()
        pos = vehicle_pose.position
        orient = vehicle_pose.orientation
        position_array = np.array([float(pos.x_val), float(pos.y_val), float(pos.z_val)])
        rotation_matrix = self.quaternion_to_rotation_matrix(orient)
        return position_array, rotation_matrix

    def quaternion_to_rotation_matrix(self, q):
        qw, qx, qy, qz = q.w_val, q.x_val, q.y_val, q.z_val
        return np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])

    def transform_to_world(self, points, position, rotation_matrix):
        points_rotated = np.dot(points, rotation_matrix.T)
        return points_rotated + position

# ---------------------------------------------------------------------------
# Grid Map: Accumulates ground (and obstacle) heights per cell.
# ---------------------------------------------------------------------------
class GridMap:
    def __init__(self, resolution):
        self.resolution = resolution
        # Store (sum, count) per cell
        self.grid = {}

    def get_grid_cell(self, x, y):
        return (round(x / self.resolution, 1), round(y / self.resolution, 1))

    def add_point(self, x, y, z, timestamp):
        cell = self.get_grid_cell(x, y)
        if cell not in self.grid:
            self.grid[cell] = [z, 1]
        else:
            self.grid[cell][0] += z
            self.grid[cell][1] += 1

    def get_height_estimate(self):
        estimates = []
        for (gx, gy), (z_sum, count) in self.grid.items():
            mean_z = z_sum / count
            estimates.append([gx * self.resolution, gy * self.resolution, mean_z])
        return np.array(estimates)

# ---------------------------------------------------------------------------
# PID Controller for Steering and Forward Motion
# ---------------------------------------------------------------------------
class PIDController:
    def __init__(self, kp, ki, kd, dt=0.1):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

# ---------------------------------------------------------------------------
# Fade risk near high-risk cells using a distance transform.
# ---------------------------------------------------------------------------
def fade_with_distance_transform(risk_grid, high_threshold=0.4, fade_scale=4.0, sigma=5.0):
    grid_max = np.nanmax(risk_grid)
    threshold_val = high_threshold * grid_max
    high_mask = risk_grid > threshold_val
    dist_map = distance_transform_edt(~high_mask)
    fade_risk   = fade_scale * np.exp(-dist_map / sigma)
    return np.maximum(risk_grid, fade_risk)
point_cloud = o3d.io.read_point_cloud('grid_point_cloud.ply')
obs_point_cloud = o3d.io.read_point_cloud('obstacle_point_cloud.ply')

# Convert the point cloud to a numpy array
points = np.asarray(point_cloud.points)
obs_point_cloud = np.asarray(obs_point_cloud.points)
# Extract X, Y, Z for ground points
ground_x_vals = points[:, 0]
ground_y_vals = points[:, 1]
ground_z_vals = points[:, 2]

# (Optional) If you have obs_point_cloud defined elsewhere:
obstacle_x_vals = obs_point_cloud[:, 0]
obstacle_y_vals = obs_point_cloud[:, 1]
obstacle_z_vals = obs_point_cloud[:, 2]
# Define the grid resolution
grid_resolution = 0.1

# Create grid edges for X and Y based on the range of ground points
x_edges = np.arange(min(ground_x_vals), max(ground_x_vals) + grid_resolution, grid_resolution)
y_edges = np.arange(min(ground_y_vals), max(ground_y_vals) + grid_resolution, grid_resolution)

# Create meshgrid for X and Y (for ground)
x_mid = (x_edges[:-1] + x_edges[1:]) / 2  # Midpoints of X bins
y_mid = (y_edges[:-1] + y_edges[1:]) / 2  # Midpoints of Y bins
X, Y = np.meshgrid(x_mid, y_mid)
Z_ground, _, _, _ = binned_statistic_2d(
    ground_x_vals, ground_y_vals, ground_z_vals, statistic='mean', bins=[x_edges, y_edges]
)
# 4. Calculate slope and step risks from ground data.
# non_nan_indices holds indices (i, j) where Z_ground is not nan.
non_nan_indices = np.argwhere(~np.isnan(Z_ground))
step_risk_grid, slope_risk_grid = calculate_combined_risks(
    Z_ground, non_nan_indices, max_height_diff=0.032, max_slope_degrees=15.0, radius=0.5
)
combined_mask = np.isnan(step_risk_grid) & np.isnan(slope_risk_grid)
masked_step_risk = np.ma.masked_array(step_risk_grid, mask=combined_mask) * 1.0
masked_slope_risk = np.ma.masked_array(slope_risk_grid, mask=combined_mask) * 3.0
sum_grid = np.ma.filled(masked_step_risk, 0) + np.ma.filled(masked_slope_risk, 0)
both_nan_mask = np.isnan(step_risk_grid) & np.isnan(slope_risk_grid)
total_risk_grid = np.where(both_nan_mask, np.nan, sum_grid)

# Incorporate obstacle risk.

obs_x_idx = np.clip(np.digitize(obs_point_cloud[:, 0], x_edges) - 1, 0, len(x_mid)-1)
obs_y_idx = np.clip(np.digitize(obs_point_cloud[:, 1], y_edges) - 1, 0, len(y_mid)-1)
total_risk_grid[obs_x_idx, obs_y_idx] = 3.0

# Apply fading and transform risk values.
total_risk_grid = fade_with_distance_transform(total_risk_grid,
                                                high_threshold=0.65,
                                                fade_scale=4.0,
                                                sigma=2.0)
# max_risk = np.nanmax(total_risk_grid)
# threshold = 0.01 * max_risk
# mask = total_risk_grid > threshold
# total_risk_grid[mask] = np.exp(total_risk_grid[mask])
total_risk_grid = interpolate_in_radius(total_risk_grid, 1.5)
masked_total_risk_grid = ma.masked_invalid(total_risk_grid)
cvar_combined_risk = compute_cvar_cellwise(masked_total_risk_grid, alpha=0.8, radius=4.0)
# cvar_combined_risk = fade_with_distance_transform(cvar_combined_risk,
#                                                 high_threshold=0.65,
#                                                 fade_scale=4.0,
#                                                 sigma=3.0)
cvar_combined_risk  = np.ma.masked_invalid(cvar_combined_risk)
max_risk = np.nanmax(cvar_combined_risk)
threshold = 0.01 * max_risk
mask = cvar_combined_risk > threshold
cvar_combined_risk[mask] = np.exp(cvar_combined_risk[mask])
# cvar_combined_risk = cvar_combined_risk.filled(0.50)

# 8. Visualization of the final CVaR risk map.
# Build a custom colormap from gray to yellow to red.
# 1) Define your five colors in order:
colors = [
    (0.5, 0.5, 0.5),  # gray
    (1.0, 1.0, 0.0),  # yellow
    (1.0, 0.65, 0.0), # orange
    (1.0, 0.0, 0.0),  # red
    # (0.0, 0.0, 0.0)   # black
]

# 2) Create a smooth colormap from them:
cmap = LinearSegmentedColormap.from_list("gray_yellow_orange_red", colors, N=256)

# 3) Plot your CVaR map with the new cmap:
plt.figure(figsize=(10,8))
plt.title("Continuous CVaR Risk Map")
plt.xlabel("Y")
plt.ylabel("X")
plt.pcolormesh(Y, X, cvar_combined_risk.T, shading='auto', cmap=cmap, alpha=0.9)
plt.colorbar(label="CVaR Risk")
plt.tight_layout()
plt.show()