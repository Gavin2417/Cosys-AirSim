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
from function2 import calculate_combined_risks,compute_cvar_cellwise
from matplotlib.colors import LinearSegmentedColormap
import numpy.ma as ma  
import heapq
def find_nearest_valid_cell(risk_grid, dest_idx):
    """Find the nearest valid and unblocked cell to the destination."""
    rows, cols = risk_grid.shape
    min_distance = float('inf')
    nearest_cell = None

    for i in range(rows):
        for j in range(cols):
            if is_unblocked(risk_grid, i, j):
                distance = calculate_h_value(i, j, dest_idx)
                if distance < min_distance:
                    min_distance = distance
                    nearest_cell = (i, j)
    
    return nearest_cell
def is_valid(row, col, grid):
    """Check if a cell is valid and within bounds."""
    return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]

def is_unblocked(grid, row, col):
    """Check if a cell is unblocked (not NaN and below a risk threshold)."""
    return not np.isnan(grid[row, col]) and grid[row, col] < 1.0

def calculate_h_value(row, col, dest):
    """Calculate the heuristic value (Euclidean distance)."""
    return np.sqrt((row - dest[0]) ** 2 + (col - dest[1]) ** 2)

def trace_path(cell_details, dest):
    """Trace the path from source to destination."""
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
    """Perform A* search on the risk map."""
    rows, cols = risk_grid.shape
    open_list = []
    heapq.heappush(open_list, (0.0, start_idx))  # (f, (row, col))
    came_from = {}
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
        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4 directions
            neighbor = (current_row + direction[0], current_col + direction[1])
            if is_valid(*neighbor, risk_grid) and is_unblocked(risk_grid, *neighbor):
                tentative_g_score = g_scores[current] + risk_grid[neighbor]

                if tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    f_scores[neighbor] = tentative_g_score + calculate_h_value(*neighbor, dest_idx)
                    heapq.heappush(open_list, (f_scores[neighbor], neighbor))
                    cell_details[neighbor] = current

    return None  # No path found
class lidarTest:
    
    def __init__(self, lidar_name, vehicle_name):
        # Connect to the AirSim simulator
        self.client = airsim.CarClient()
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
                return None
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
            return None

    def get_vehicle_pose(self):
        # Get the pose (position and orientation) of the vehicle in world coordinates
        vehicle_pose = self.client.simGetVehiclePose()
        position = vehicle_pose.position
        orientation = vehicle_pose.orientation

        # Convert position to a numpy array with float values
        position_array = np.array([float(position.x_val), float(position.y_val), float(position.z_val)])

        # Convert quaternion orientation to a rotation matrix
        q = orientation
        rotation_matrix = self.quaternion_to_rotation_matrix(q)

        return position_array, rotation_matrix

    def quaternion_to_rotation_matrix(self, q):
        # Convert quaternion to a 3x3 rotation matrix
        qw, qx, qy, qz = q.w_val, q.x_val, q.y_val, q.z_val

        # Create a 4x4 transformation matrix from the quaternion
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

class GridMap:
    def __init__(self, resolution):
        self.resolution = resolution
        self.grid = {}

    def get_grid_cell(self, x, y):
        grid_x = round(int(x / self.resolution), 1)
        grid_y = round(int(y / self.resolution), 1)
        return grid_x, grid_y

    def add_point(self, x, y, z, timestamp):
        # Get grid cell for the point
        cell = self.get_grid_cell(x, y)
        
        if cell not in self.grid:
            # Initialize a grid cell list
            self.grid[cell] = []

        # Append the point to the grid cell
        self.grid[cell].append(z)

    def get_height_estimate(self):
        # Return the estimated height (average Z) for each cell
        estimated_points = []
        for cell, z_values in self.grid.items():
            # Average Z values in the grid cell
            avg_z = np.mean(z_values)
            x, y = cell
            estimated_points.append([x * self.resolution, y * self.resolution, avg_z])
        return np.array(estimated_points)
# Main
if __name__ == "__main__":
    # Initialize Lidar test
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    grid_map_ground = GridMap(resolution=0.1)
    grid_map_obstacle = GridMap(resolution=0.1)

    # Initialize ground segmentation object with default or config file
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
    config_path = f"{BASE_DIR}/../assets/config.toml"
    
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, using default parameters")
        groundseg = ground_seg()
    else:
        groundseg = ground_seg(config_path)

    # Initialize visualizer
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name='Lidar Visualization', width=800, height=600)
    fig, ax = plt.subplots()  # No 'projection=3d'
    plt.ion()  # Enable interactive mode
    colorbar = None
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

                for i, point in enumerate(points_world):
                    x, y, z = point
                    if label[i] == 1:
                        grid_map_ground.add_point(x, y, z, timestamp)
                    elif z > -0.2:
                        grid_map_obstacle.add_point(x, y, z, timestamp)

                ground_points = grid_map_ground.get_height_estimate()
                obstacle_points = grid_map_obstacle.get_height_estimate()
                                
                # # Convert the point clouds to numpy arrays
                # ground_points = np.asarray(ground_point_cloud.points)
                # obstacle_points = np.asarray(obstacle_point_cloud.points)

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
                start_point = position[:2]
                destination_point = np.array([15, -10])  # Example destination

               
                start_idx = (
                    np.digitize(start_point[0], x_edges) - 1,
                    np.digitize(start_point[1], y_edges) - 1,
                )
                dest_idx = (
                    np.digitize(destination_point[0], x_edges) - 1,
                    np.digitize(destination_point[1], y_edges) - 1,
                )

                # Check if the destination point is valid; if not, find the nearest valid cell
                if not is_valid(*dest_idx, cvar_combined_risk) or not is_unblocked(cvar_combined_risk, *dest_idx):
                    print("Destination point is blocked. Finding the nearest valid cell...")
                    dest_idx = find_nearest_valid_cell(cvar_combined_risk, dest_idx)

                if dest_idx is None:
                    print("No valid destination point found on the risk map!")
                    continue

                # Run A* search
                path = a_star_search(cvar_combined_risk, start_idx, dest_idx)

                # Visualization
                colors = [(0.5, 0.5, 0.5), (1, 1, 0), (1, 0, 0)]
                cmap = LinearSegmentedColormap.from_list("gray_yellow_red", colors)

                ax.clear()
                c = ax.pcolormesh(X, Y, cvar_combined_risk.T, shading='auto', cmap=cmap, alpha=0.7)
                if colorbar is None:
                    colorbar = fig.colorbar(c, ax=ax, label='Risk Value (0=zero risk, 1=high risk)')
                else:
                    colorbar.update_normal(c)

                if path:
                    path_x = [x_mid[p[0]] for p in path]
                    path_y = [y_mid[p[1]] for p in path]
                    ax.plot(path_x, path_y, color="blue", linewidth=2, label="A* Path")
                else:
                    print("No path found!")

                ax.scatter(start_point[0], start_point[1], color="green", label="Start", zorder=5)
                ax.scatter(destination_point[0], destination_point[1], color="red", label="Destination", zorder=5)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.legend()
                ax.set_title("A* Path on Risk Map")

                plt.draw()
                plt.pause(0.1)
    finally:
        plt.ioff()
        plt.show()