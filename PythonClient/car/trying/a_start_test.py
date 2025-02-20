import cosysairsim as airsim
import numpy as np
import open3d as o3d
import time
from linefit import ground_seg
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
from function2 import calculate_combined_risks, compute_cvar_cellwise
from matplotlib.colors import LinearSegmentedColormap
import numpy.ma as ma
from scipy.spatial import cKDTree
import heapq
import math  # For trigonometry

# ---------------------------------------------------------------------------
# PID Controller for Steering
# ---------------------------------------------------------------------------
class PIDController:
    def __init__(self, kp, ki, kd, dt=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

# Instantiate a PID controller for steering.
steering_pid = PIDController(kp=0.75, ki=0.01, kd=0.25, dt=0.1)

# ---------------------------------------------------------------------------
# Interpolation: Fill in missing (NaN) grid cells using nearby valid cells.
# ---------------------------------------------------------------------------
def interpolate_in_radius(grid, radius):
    """
    Interpolates NaN values in a grid using valid points within a specified radius.
    """
    valid_points = ~np.isnan(grid)
    valid_coords = np.column_stack(np.where(valid_points))
    valid_values = grid[valid_points]

    # Create KDTree for efficient neighbor search
    tree = cKDTree(valid_coords)
    nan_coords = np.column_stack(np.where(np.isnan(grid)))

    # Iterate through each NaN point
    for idx, coord in enumerate(nan_coords):
        neighbors = tree.query_ball_point(coord, radius)
        if neighbors:
            weights = []
            weighted_values = []
            for neighbor_idx in neighbors:
                neighbor_coord = valid_coords[neighbor_idx]
                value = valid_values[neighbor_idx]
                distance = np.linalg.norm(coord - neighbor_coord)
                weight = 1 / (distance + 1e-6)  # Avoid division by zero
                weights.append(weight)
                weighted_values.append(weight * value)
            grid[coord[0], coord[1]] = np.sum(weighted_values) / np.sum(weights)
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

def is_unblocked(grid, row, col):
    return (not np.isnan(grid[row, col])) and (grid[row, col] < 1.0)

def calculate_h_value(row, col, dest):
    return np.sqrt((row - dest[0]) ** 2 + (col - dest[1]) ** 2)

def trace_path(cell_details, dest):
    path = []
    row, col = dest
    while True:
        path.append((row, col))
        parent_row, parent_col = cell_details[row, col]
        if (row, col) == (parent_row, parent_col):
            break
        row, col = parent_row, parent_col
    path.reverse()
    return path

def a_star_search(risk_grid, start_idx, dest_idx):
    rows, cols = risk_grid.shape
    open_list = []
    heapq.heappush(open_list, (0.0, start_idx))
    g_scores = np.full((rows, cols), float('inf'))
    g_scores[start_idx] = 0
    f_scores = np.full((rows, cols), float('inf'))
    f_scores[start_idx] = calculate_h_value(*start_idx, dest_idx)
    cell_details = np.full((rows, cols), None, dtype=object)
    for i in range(rows):
        for j in range(cols):
            cell_details[i, j] = (i, j)
    
    while open_list:
        _, current = heapq.heappop(open_list)
        if current == dest_idx:
            return trace_path(cell_details, dest_idx)
        current_row, current_col = current
        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current_row + direction[0], current_col + direction[1])
            if is_valid(neighbor[0], neighbor[1], risk_grid) and is_unblocked(risk_grid, neighbor[0], neighbor[1]):
                tentative_g_score = g_scores[current] + risk_grid[neighbor]
                if tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    f_scores[neighbor] = tentative_g_score + calculate_h_value(neighbor[0], neighbor[1], dest_idx)
                    heapq.heappush(open_list, (f_scores[neighbor], neighbor))
                    cell_details[neighbor] = current
    return None  # No path found

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
        # Not enough points to smooth; return original path.
        return path

    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2
    smoothed = []
    for i in range(n_points):
        start_idx = max(0, i - half_window)
        end_idx = min(n_points, i + half_window + 1)
        window_average = np.mean(path[start_idx:end_idx], axis=0)
        smoothed.append(window_average)
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
            if len(lidarData.point_cloud) < 2:
                self.lastlidarTimeStamp = lidarData.time_stamp
                return None, None
            else:
                self.lastlidarTimeStamp = lidarData.time_stamp
                points = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
                num_dims = 5 if gpulidar else 3
                points = np.reshape(points, (int(points.shape[0] / num_dims), num_dims))
                if not gpulidar:
                    points = points * np.array([1, -1, 1])
                return points, lidarData.time_stamp
        else:
            return None, None

    def get_vehicle_pose(self):
        vehicle_pose = self.client.simGetVehiclePose()
        position = vehicle_pose.position
        orientation = vehicle_pose.orientation
        position_array = np.array([float(position.x_val), float(position.y_val), float(position.z_val)])
        rotation_matrix = self.quaternion_to_rotation_matrix(orientation)
        return position_array, rotation_matrix

    def quaternion_to_rotation_matrix(self, q):
        qw, qx, qy, qz = q.w_val, q.x_val, q.y_val, q.z_val
        rotation_matrix = np.array([
            [1.0 - 2.0*qy*qy - 2.0*qz*qz, 2.0*qx*qy - 2.0*qz*qw, 2.0*qx*qz + 2.0*qy*qw],
            [2.0*qx*qy + 2.0*qz*qw, 1.0 - 2.0*qx*qx - 2.0*qz*qz, 2.0*qy*qz - 2.0*qx*qw],
            [2.0*qx*qz - 2.0*qy*qw, 2.0*qy*qz + 2.0*qx*qw, 1.0 - 2.0*qx*qx - 2.0*qy*qy],
        ])
        return rotation_matrix

    def transform_to_world(self, points, position, rotation_matrix):
        points_rotated = np.dot(points, rotation_matrix.T)
        points_in_world = points_rotated + position
        return points_in_world

# ---------------------------------------------------------------------------
# Grid Map: Accumulates ground (and obstacle) heights per cell.
# ---------------------------------------------------------------------------
class GridMap:
    def __init__(self, resolution):
        self.resolution = resolution
        self.grid = {}

    def get_grid_cell(self, x, y):
        grid_x = round(int(x / self.resolution), 1)
        grid_y = round(int(y / self.resolution), 1)
        return grid_x, grid_y

    def add_point(self, x, y, z, timestamp):
        cell = self.get_grid_cell(x, y)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(z)

    def get_height_estimate(self):
        estimated_points = []
        for cell, z_values in self.grid.items():
            avg_z = np.mean(z_values)
            x, y = cell
            estimated_points.append([x * self.resolution, y * self.resolution, avg_z])
        return np.array(estimated_points)

# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    lidar_test.client.enableApiControl(True, 'CPHusky')
        
    grid_map_ground = GridMap(resolution=0.1)
    grid_map_obstacle = GridMap(resolution=0.1)

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
    config_path = f"{BASE_DIR}/../assets/config.toml"
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, using default parameters")
        groundseg = ground_seg()
    else:
        groundseg = ground_seg(config_path)

    fig, ax = plt.subplots()
    plt.ion()
    colorbar = None
    current_path = None

    lidar_test.client.setCarControls(airsim.CarControls(throttle=0.03, steering=0), 'CPHusky')
    try:
        while True:
            point_cloud_data, timestamp = lidar_test.get_data(gpulidar=True)
            if point_cloud_data is not None:
                points = np.array(point_cloud_data[:, :3], dtype=np.float64)
                points = points[np.linalg.norm(points, axis=1) > 0.6]

                position, rotation_matrix = lidar_test.get_vehicle_pose()
                points_world = lidar_test.transform_to_world(points, position, rotation_matrix)
                points_world[:, 2] = -points_world[:, 2]
                label = np.array(groundseg.run(points_world))

                # Populate grid maps.
                for i, point in enumerate(points_world):
                    x, y, z = point
                    if label[i] == 1:
                        grid_map_ground.add_point(x, y, z, timestamp)
                    elif z > -0.2:
                        grid_map_obstacle.add_point(x, y, z, timestamp)

                ground_points = grid_map_ground.get_height_estimate()
                obstacle_points = grid_map_obstacle.get_height_estimate()

                vehicle_x, vehicle_y = position[0], position[1]
                center = np.array([vehicle_x, vehicle_y])
                ground_points = filter_points_by_radius(ground_points, center, 15)
                obstacle_points = filter_points_by_radius(obstacle_points, center, 15)

                ground_x_vals = ground_points[:, 0]
                ground_y_vals = ground_points[:, 1]
                ground_z_vals = ground_points[:, 2]

                obstacle_x_vals = obstacle_points[:, 0]
                obstacle_y_vals = obstacle_points[:, 1]
                obstacle_z_vals = obstacle_points[:, 2]

                # Create a fixed grid covering start and destination.
                start_world = np.array([vehicle_x, vehicle_y])
                destination_point = np.array([10, -5])
                margin = 10.0
                min_x = min(start_world[0], destination_point[0]) - margin
                max_x = max(start_world[0], destination_point[0]) + margin
                min_y = min(start_world[1], destination_point[1]) - margin
                max_y = max(start_world[1], destination_point[1]) + margin

                grid_resolution = 0.1
                x_edges = np.arange(min_x, max_x + grid_resolution, grid_resolution)
                y_edges = np.arange(min_y, max_y + grid_resolution, grid_resolution)
                x_mid = (x_edges[:-1] + x_edges[1:]) / 2
                y_mid = (y_edges[:-1] + y_edges[1:]) / 2
                X, Y = np.meshgrid(x_mid, y_mid)

                Z_ground = np.full((len(x_mid), len(y_mid)), np.nan)
                for i in range(len(ground_x_vals)):
                    x_idx = np.digitize(ground_x_vals[i], x_edges) - 1
                    y_idx = np.digitize(ground_y_vals[i], y_edges) - 1
                    if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
                        Z_ground[x_idx, y_idx] = ground_z_vals[i]

                non_nan_indices = np.argwhere(~np.isnan(Z_ground))
                step_risk_grid, slope_risk_grid = calculate_combined_risks(
                    Z_ground, non_nan_indices, max_height_diff=0.05, max_slope_degrees=30.0, radius=0.5
                )
                combined_mask = np.isnan(step_risk_grid) & np.isnan(slope_risk_grid)
                masked_step_risk = np.ma.masked_array(step_risk_grid, mask=combined_mask)
                masked_slope_risk = np.ma.masked_array(slope_risk_grid, mask=combined_mask)
                total_risk_grid = np.ma.mean([masked_step_risk, masked_slope_risk], axis=0).filled(np.nan)

                for i in range(len(obstacle_x_vals)):
                    x_idx = np.digitize(obstacle_x_vals[i], x_edges) - 1
                    y_idx = np.digitize(obstacle_y_vals[i], y_edges) - 1
                    if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
                        total_risk_grid[x_idx, y_idx] = 1.0

                interpolation_radius = 1.5
                total_risk_grid = interpolate_in_radius(total_risk_grid, interpolation_radius)
                masked_total_risk_grid = ma.masked_invalid(total_risk_grid)
                cvar_combined_risk = compute_cvar_cellwise(masked_total_risk_grid)

                distance_from_vehicle = np.sqrt((X - vehicle_x)**2 + (Y - vehicle_y)**2)
                cvar_combined_risk[distance_from_vehicle.T > 13.0] = np.nan

                # A* Path Planning.
                start_idx = (np.digitize(vehicle_x, x_edges) - 1, np.digitize(vehicle_y, y_edges) - 1)
                destination_point = np.array([10, -5])
                dest_idx = (np.digitize(destination_point[0], x_edges) - 1, np.digitize(destination_point[1], y_edges) - 1)
                dest_idx = (min(max(dest_idx[0], 0), len(x_mid) - 1), min(max(dest_idx[1], 0), len(y_mid) - 1))
                
                
                if current_path is None:
                    if np.isnan(cvar_combined_risk[start_idx[0], start_idx[1]]):
                        current_path = None
                    else:
                        current_path = a_star_search(cvar_combined_risk, start_idx, dest_idx)
                        if current_path is None:
                            valid_indices = np.argwhere(~np.isnan(cvar_combined_risk))
                            if valid_indices.size > 0:
                                candidate_centers = np.column_stack((x_mid[valid_indices[:, 0]], y_mid[valid_indices[:, 1]]))
                                candidate_distances = np.linalg.norm(candidate_centers - destination_point, axis=1)
                                best_candidate = valid_indices[np.argmin(candidate_distances)]
                                dest_idx = tuple(best_candidate)
                                current_path = a_star_search(cvar_combined_risk, start_idx, dest_idx)
                            else:
                                current_path = None
                else:
                    path_valid = True
                    for cell in current_path:
                        if np.isnan(cvar_combined_risk[cell[0], cell[1]]) or cvar_combined_risk[cell[0], cell[1]] >= 1.0:
                            path_valid = False
                            break
                    if not path_valid:
                        current_path = None

                # Visualization.
                colors = [(0.5, 0.5, 0.5), (1, 1, 0), (1, 0, 0)]
                cmap = LinearSegmentedColormap.from_list("gray_yellow_red", colors)
                ax.clear()
                c = ax.pcolormesh(X, Y, cvar_combined_risk.T, shading='auto', cmap=cmap, alpha=0.7)
                if colorbar is None:
                    colorbar = fig.colorbar(c, ax=ax, label='Risk Value (0=zero risk, 1=risky)')
                else:
                    colorbar.update_normal(c)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title('Risk Visualization with A* Path')
                # lidar_test.client.setCarControls(airsim.CarControls(throttle=0.05, steering=0), 'CPHusky')
                # If a path exists, overlay it.
                
                if current_path:
                    print("Path found!")
                    raw_path = np.array([[x_mid[cell[0]], y_mid[cell[1]]] for cell in current_path])
                    smoothed_path = smooth_path(raw_path, window_size=5)
                    ax.plot(smoothed_path[:, 0], smoothed_path[:, 1], color="blue", linewidth=2, label="Smoothed A* Path")
                    
                    # -------------------------------
                    # PID Control: Follow the Path
                    # -------------------------------
                    # Choose a lookahead waypoint along the smoothed path.
                    lookahead_idx = min(5, len(smoothed_path) - 1)
                    target_point = smoothed_path[lookahead_idx]

                    # Compute desired heading.
                    desired_heading = math.atan2(target_point[1] - vehicle_y,
                                                 target_point[0] - vehicle_x)
                    # Compute current heading from the rotation matrix.
                    current_heading = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                    # Compute heading error (wrapped to [-pi, pi]).
                    error = desired_heading - current_heading
                    error = (error + math.pi) % (2 * math.pi) - math.pi
                    # Compute steering command via PID.
                    steering1 = steering_pid.compute(error)
                    # Clamp steering to [-1, 1].
                    steering1 = max(min(steering1, 1.0), -1.0)

                    # For this example, we set a constant throttle.
                    throttle1 = 0.025
                    
                    # Create and send control command.
                    # car_controls = airsim.CarControls()
                    # car_controls.throttle = throttle
                    # car_controls.steering = steering
                    lidar_test.client.setCarControls(airsim.CarControls(throttle=throttle1, steering=steering1), lidar_test.vehicleName)
                    print("Fallback command: moving forward", lidar_test.vehicleName)
                    # -------------------------------
                else:
                    lidar_test.client.setCarControls(airsim.CarControls(throttle=0.029, steering=0), 'CPHusky')
                # Mark the start and destination.
                ax.scatter(vehicle_x, vehicle_y, color="green", label="Start", zorder=5)
                ax.scatter(destination_point[0], destination_point[1], color="red", label="Destination", zorder=5)
                ax.legend()

                plt.draw()
                plt.pause(0.1)

    finally:
        plt.ioff()
        plt.show()
