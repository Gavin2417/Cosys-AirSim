# import setup_path
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
    
    Parameters:
        path (array-like): An array of points [[x1, y1], [x2, y2], ...].
        window_size (int): The number of points to average over (should be odd).
    
    Returns:
        np.ndarray: Smoothed path as an array of points.
    """
    path = np.array(path)
    n_points = len(path)
    if n_points < window_size:
        # Not enough points to smooth; return original path.
        return path

    # If window_size is even, increment it by 1 to ensure symmetry.
    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2
    smoothed = []
    for i in range(n_points):
        # Define window bounds (handling the boundaries)
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
        # Connect to the AirSim simulator
        self.client = airsim.CarClient(ip="100.123.124.47")
        self.client.confirmConnection()
        self.vehicleName = vehicle_name
        self.lidarName = lidar_name
        self.lastlidarTimeStamp = 0

    def get_data(self, gpulidar):
        # Get lidar data from AirSim
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
                # Process lidar point cloud data
                points = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
                num_dims = 5 if gpulidar else 3
                points = np.reshape(points, (int(points.shape[0] / num_dims), num_dims))
                if not gpulidar:
                    points = points * np.array([1, -1, 1])  # Adjust for AirSim coordinates
                return points, lidarData.time_stamp  # Return timestamp with data
        else:
            return None, None

    def get_vehicle_pose(self):
        # Get the pose (position and orientation) of the vehicle in world coordinates
        vehicle_pose = self.client.simGetVehiclePose()
        position = vehicle_pose.position
        orientation = vehicle_pose.orientation
        position_array = np.array([float(position.x_val), float(position.y_val), float(position.z_val)])
        q = orientation
        rotation_matrix = self.quaternion_to_rotation_matrix(q)
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
        # Apply rotation first, then apply translation (position)
        points_rotated = np.dot(points, rotation_matrix.T)  # Rotate points
        points_in_world = points_rotated + position  # Translate points to world coordinates
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
    # Initialize Lidar test and grid maps.
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    lidar_test.client.enableApiControl(True, 'CPHusky')
    grid_map_ground = GridMap(resolution=0.1)
    grid_map_obstacle = GridMap(resolution=0.1)

    # Initialize ground segmentation object with default or config file.
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
    config_path = f"{BASE_DIR}/../assets/config.toml"
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, using default parameters")
        groundseg = ground_seg()
    else:
        groundseg = ground_seg(config_path)

    # Initialize visualizer.
    fig, ax = plt.subplots()  # No 'projection=3d'
    plt.ion()  # Enable interactive mode
    colorbar = None
    path = None
    temp_dest = None
    temp_path = None
    try:
        while True:
            point_cloud_data, timestamp = lidar_test.get_data(gpulidar=True)
            if point_cloud_data is not None:
                points = np.array(point_cloud_data[:, :3], dtype=np.float64)
                points = points[np.linalg.norm(points, axis=1) > 0.6]

                position, rotation_matrix = lidar_test.get_vehicle_pose()
                points_world = lidar_test.transform_to_world(points, position, rotation_matrix)
                points_world[:, 2] = -points_world[:, 2]  # Flip Z-axis if needed
                label = np.array(groundseg.run(points_world))

                # Populate grid maps.
                for i, point in enumerate(points_world):
                    x, y, z = point
                    if label[i] == 1:
                        grid_map_ground.add_point(x, y, z, timestamp)
                    elif z > -position[2]:
                        grid_map_obstacle.add_point(x, y, z, timestamp)
                    else:
                        grid_map_ground.add_point(x, y, z, timestamp)

                # Retrieve height estimates.
                ground_points = grid_map_ground.get_height_estimate()
                obstacle_points = grid_map_obstacle.get_height_estimate()

                # Apply radius filtering (keep points within 15 units of the vehicle).
                vehicle_x, vehicle_y = position[0], position[1]
                center = np.array([vehicle_x, vehicle_y])
                radius = 15
                ground_points = filter_points_by_radius(ground_points, center, radius)

                # Extract X, Y, Z for ground and obstacle points.
                ground_x_vals = ground_points[:, 0]
                ground_y_vals = ground_points[:, 1]
                ground_z_vals = ground_points[:, 2]

                # Define the grid resolution and create grid edges.
                grid_resolution = 0.1
                margin = 1
                start_world = np.array([-1, 0])  # current vehicle position
                destination_point = np.array([10, -5])     
                # Compute the bounding box limits
                min_x = min(start_world[0], destination_point[0]) - margin
                max_x = max(start_world[0], destination_point[0]) + margin
                min_y = min(start_world[1], destination_point[1]) - margin
                max_y = max(start_world[1], destination_point[1]) + margin

                # Define grid resolution (cell size in meters)
                grid_resolution = 0.1

                # Create fixed grid edges and midpoints
                x_edges = np.arange(min_x, max_x + grid_resolution, grid_resolution)
                y_edges = np.arange(min_y, max_y + grid_resolution, grid_resolution)
                x_mid = (x_edges[:-1] + x_edges[1:]) / 2  # Fixed midpoints of X bins.
                y_mid = (y_edges[:-1] + y_edges[1:]) / 2  # Fixed midpoints of Y bins.
                X, Y = np.meshgrid(x_mid, y_mid)

                # Initialize a Z grid for ground points.
                Z_ground = np.full((len(x_mid), len(y_mid)), np.nan)
                for i in range(len(ground_x_vals)):
                    x_idx = np.digitize(ground_x_vals[i], x_edges) - 1
                    y_idx = np.digitize(ground_y_vals[i], y_edges) - 1
                    if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
                        Z_ground[x_idx, y_idx] = ground_z_vals[i]

                # Calculate the combined step and slope risk grids.
                non_nan_indices = np.argwhere(~np.isnan(Z_ground))
                step_risk_grid, slope_risk_grid = calculate_combined_risks(
                    Z_ground, non_nan_indices, max_height_diff=0.05, max_slope_degrees=30.0, radius=0.5
                )
                combined_mask = np.isnan(step_risk_grid) & np.isnan(slope_risk_grid)
                masked_step_risk = np.ma.masked_array(step_risk_grid, mask=combined_mask)
                masked_slope_risk = np.ma.masked_array(slope_risk_grid, mask=combined_mask)
                total_risk_grid = np.ma.mean([masked_step_risk, masked_slope_risk], axis=0).filled(np.nan)

                if obstacle_points.size != 0:
                    obstacle_points = filter_points_by_radius(obstacle_points, center, radius)
                    
                    obstacle_x_vals = obstacle_points[:, 0]
                    obstacle_y_vals = obstacle_points[:, 1]
                    obstacle_z_vals = obstacle_points[:, 2]

                    # Add obstacle points to the risk grid
                    for i in range(len(obstacle_x_vals)):
                        x_idx = np.digitize(obstacle_x_vals[i], x_edges) - 1
                        y_idx = np.digitize(obstacle_y_vals[i], y_edges) - 1
                        if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
                            total_risk_grid[x_idx, y_idx] = 1.0  # Mark obstacles as high risk

                # Interpolate missing values in the risk grid.
                interpolation_radius = 1.5
                total_risk_grid = interpolate_in_radius(total_risk_grid, interpolation_radius)
                masked_total_risk_grid = ma.masked_invalid(total_risk_grid)
                cvar_combined_risk = compute_cvar_cellwise(masked_total_risk_grid, alpha=0.8)

                # Optionally mask cells far from the vehicle.
                distance_from_vehicle = np.sqrt((X - vehicle_x)**2 + (Y - vehicle_y)**2)
                cvar_combined_risk[distance_from_vehicle.T > 13.0] = np.nan

                # Convert vehicle position to grid indices.
                start_idx = (np.digitize(vehicle_x, x_edges) - 1, np.digitize(vehicle_y, y_edges) - 1)
                destination_point = np.array([10, -5])
                dest_idx = (np.digitize(destination_point[0], x_edges) - 1, np.digitize(destination_point[1], y_edges) - 1)
   
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

                
                # convert to  greid points
                manual_path = [
                    (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
                    (6, 0), (7, 0), (8, 0), (8,-1), (8,-2), (8,-3), (8,-4), (8,-5), (9,-5), (10,-5), (10,-6)
                ]
                temp_i = []
                for pt in manual_path:
                    x_idx = np.digitize(pt[0], x_edges) - 1
                    y_idx = np.digitize(pt[1], y_edges) - 1
                    # Ensure indices remain within the bounds of x_mid and y_mid.
                    x_idx = np.clip(x_idx, 0, len(x_mid) - 1)
                    y_idx = np.clip(y_idx, 0, len(y_mid) - 1)
                    temp_i.append((x_idx, y_idx))
                # Convert manual path grid coordinates to world coordinates.
                # Here, x_mid and y_mid are assumed to be 1D arrays representing the midpoints of grid cells.
                manual_path_world = np.array([[x_mid[x_idx], y_mid[y_idx]] for x_idx, y_idx in temp_i])

                # Optionally, smooth the manual path.
                smoothed_manual_path = smooth_path(manual_path_world, window_size=5)

                # Plot the manual path.
                ax.plot(smoothed_manual_path[:, 0], smoothed_manual_path[:, 1],
                        color="magenta", linestyle="--", linewidth=2, label="Manual Path")

                # Mark the start and destination.
                ax.scatter(vehicle_x, vehicle_y, color="green", label="Start", zorder=5)
                ax.scatter(destination_point[0], destination_point[1], color="red", label="Destination", zorder=5)
                ax.legend()

                plt.draw()
                plt.pause(0.1)

    finally:
        plt.ioff()
        plt.show()

        manual_path = [
    (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),  # Move straight along X-axis
    (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),
    (10, 0), (10, -1), (10, -2), (10, -3), (10, -4), (10, -5)  # Turn left and reach destination
]

