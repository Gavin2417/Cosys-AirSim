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

###############################################
# A* Path Planning Functions
###############################################
def world_to_grid(x, y, x_edges, y_edges):
    """
    Convert world coordinates (x, y) into grid indices.
    """
    i = np.digitize(x, x_edges) - 1
    j = np.digitize(y, y_edges) - 1
    return int(i), int(j)

def a_star_planning(risk_grid, start_idx, goal_idx, grid_resolution):
    """
    A* planner on a 2D grid using the risk grid as cost.
    
    Parameters:
        risk_grid (2D ndarray): Grid with risk values.
        start_idx (tuple): (i, j) starting cell indices.
        goal_idx (tuple): (i, j) goal cell indices.
        grid_resolution (float): Size of a grid cell.
    
    Returns:
        List of (i,j) grid indices representing the path, or None if no path is found.
    """
    # 8-connected neighbor moves
    neighbor_offsets = [(-1,  0), (1,  0), (0, -1), (0,  1),
                        (-1, -1), (-1,  1), (1, -1), (1,  1)]
    
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    
    open_set = []
    # Heap items: (priority, cost_so_far, current_index, parent)
    heapq.heappush(open_set, (heuristic(start_idx, goal_idx), 0, start_idx, None))
    
    came_from = {}
    cost_so_far = {start_idx: 0}
    
    while open_set:
        current_priority, current_cost, current, parent = heapq.heappop(open_set)
        if current not in came_from:
            came_from[current] = parent

        if current == goal_idx:
            # Reconstruct path
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path
        
        for offset in neighbor_offsets:
            neighbor = (current[0] + offset[0], current[1] + offset[1])
            # Check bounds
            if (neighbor[0] < 0 or neighbor[0] >= risk_grid.shape[0] or
                neighbor[1] < 0 or neighbor[1] >= risk_grid.shape[1]):
                continue

            # Cost for moving: movement cost scaled by grid_resolution (diagonals cost more)
            move_cost = np.linalg.norm(np.array(offset)) * grid_resolution
            
            # Use the risk value as additional cost. If a cell is unknown, assume high risk.
            risk_cost = risk_grid[neighbor[0], neighbor[1]]
            if np.isnan(risk_cost):
                risk_cost = 1.0

            new_cost = current_cost + move_cost + risk_cost
            
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal_idx)
                heapq.heappush(open_set, (priority, new_cost, neighbor, current))
    
    return None  # No path found

###############################################
# LiDAR and GridMap Classes
###############################################
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
        rotation_matrix = self.quaternion_to_rotation_matrix(orientation)

        return position_array, rotation_matrix

    def quaternion_to_rotation_matrix(self, q):
        # Convert quaternion to a 3x3 rotation matrix
        qw, qx, qy, qz = q.w_val, q.x_val, q.y_val, q.z_val

        rotation_matrix = np.array([
            [1.0 - 2.0*qy*qy - 2.0*qz*qz, 2.0*qx*qy - 2.0*qz*qw,     2.0*qx*qz + 2.0*qy*qw],
            [2.0*qx*qy + 2.0*qz*qw,       1.0 - 2.0*qx*qx - 2.0*qz*qz, 2.0*qy*qz - 2.0*qx*qw],
            [2.0*qx*qz - 2.0*qy*qw,       2.0*qy*qz + 2.0*qx*qw,     1.0 - 2.0*qx*qx - 2.0*qy*qy],
        ])
        return rotation_matrix

    def transform_to_world(self, points, position, rotation_matrix):
        # Apply rotation first, then apply translation (position)
        points_rotated = np.dot(points, rotation_matrix.T)  # Rotate points
        points_in_world = points_rotated + position          # Translate points
        return points_in_world

def interpolate_in_radius(grid, radius):
    """
    Interpolates NaN values in a grid using valid points within a specified radius.
    
    Parameters:
        grid (ndarray): 2D grid with NaN values to interpolate.
        radius (float): Radius within which to search for valid points.
    
    Returns:
        ndarray: Grid with interpolated values.
    """
    valid_points = ~np.isnan(grid)
    valid_coords = np.column_stack(np.where(valid_points))
    valid_values = grid[valid_points]

    tree = cKDTree(valid_coords)
    nan_coords = np.column_stack(np.where(np.isnan(grid)))

    for coord in nan_coords:
        neighbors = tree.query_ball_point(coord, radius)
        if neighbors:
            weights = []
            weighted_values = []
            for neighbor_idx in neighbors:
                neighbor_coord = valid_coords[neighbor_idx]
                value = valid_values[neighbor_idx]
                distance = np.linalg.norm(coord - neighbor_coord)
                weight = 1 / (distance + 1e-6)
                weights.append(weight)
                weighted_values.append(weight * value)
            grid[coord[0], coord[1]] = np.sum(weighted_values) / np.sum(weights)

    return grid

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

###############################################
# Main
###############################################
if __name__ == "__main__":
    # Initialize LiDAR test and grid maps
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    grid_map_ground = GridMap(resolution=0.1)
    grid_map_obstacle = GridMap(resolution=0.1)

    # Initialize ground segmentation object
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
    config_path = f"{BASE_DIR}/../assets/config.toml"
    
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, using default parameters")
        groundseg = ground_seg()
    else:
        groundseg = ground_seg(config_path)

    # Set up visualization (using matplotlib)
    fig, ax = plt.subplots()
    plt.ion()  # Interactive mode enabled
    colorbar = None

    # Define the final goal in world coordinates
    final_goal_world = [10, -5]
    
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

                # Separate ground and obstacle points
                for i, point in enumerate(points_world):
                    x, y, z = point
                    if label[i] == 1:
                        grid_map_ground.add_point(x, y, z, timestamp)
                    elif z > -0.2:
                        grid_map_obstacle.add_point(x, y, z, timestamp)

                ground_points = grid_map_ground.get_height_estimate()
                obstacle_points = grid_map_obstacle.get_height_estimate()
                
                # Extract X, Y, Z values for ground points
                ground_x_vals = ground_points[:, 0]
                ground_y_vals = ground_points[:, 1]
                ground_z_vals = ground_points[:, 2]

                obstacle_x_vals = obstacle_points[:, 0]
                obstacle_y_vals = obstacle_points[:, 1]
                obstacle_z_vals = obstacle_points[:, 2]

                # Define grid resolution and edges based on ground data
                grid_resolution = 0.1
                x_edges = np.arange(min(ground_x_vals), max(ground_x_vals) + grid_resolution, grid_resolution)
                y_edges = np.arange(min(ground_y_vals), max(ground_y_vals) + grid_resolution, grid_resolution)

                # Create meshgrid for visualization (note: meshgrid returns Y,X order)
                x_mid = (x_edges[:-1] + x_edges[1:]) / 2  # midpoints of X bins
                y_mid = (y_edges[:-1] + y_edges[1:]) / 2  # midpoints of Y bins
                X, Y = np.meshgrid(x_mid, y_mid)

                # Initialize Z grid for ground points
                Z_ground = np.full((len(x_mid), len(y_mid)), np.nan)
                for i in range(len(ground_x_vals)):
                    x_idx = np.digitize(ground_x_vals[i], x_edges) - 1
                    y_idx = np.digitize(ground_y_vals[i], y_edges) - 1
                    if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
                        Z_ground[x_idx, y_idx] = ground_z_vals[i]

                non_nan_indices = np.argwhere(~np.isnan(Z_ground))

                # Calculate combined step and slope risk grids
                step_risk_grid, slope_risk_grid = calculate_combined_risks(Z_ground, non_nan_indices,
                                                                           max_height_diff=0.05,
                                                                           max_slope_degrees=30.0,
                                                                           radius=0.5)

                combined_mask = np.isnan(step_risk_grid) & np.isnan(slope_risk_grid)
                masked_step_risk = np.ma.masked_array(step_risk_grid, mask=combined_mask)
                masked_slope_risk = np.ma.masked_array(slope_risk_grid, mask=combined_mask)

                total_risk_grid = np.ma.mean([masked_step_risk, masked_slope_risk], axis=0).filled(np.nan)

                # Mark obstacles in the risk grid as high risk
                for i in range(len(obstacle_x_vals)):
                    x_idx = np.digitize(obstacle_x_vals[i], x_edges) - 1
                    y_idx = np.digitize(obstacle_y_vals[i], y_edges) - 1
                    if 0 <= x_idx < len(x_mid) and 0 <= y_idx < len(y_mid):
                        total_risk_grid[x_idx, y_idx] = 1.0

                # Interpolate missing values in the risk grid
                interpolation_radius = 1.5
                total_risk_grid = interpolate_in_radius(total_risk_grid, interpolation_radius)

                # Mask NaN values for visualization purposes
                masked_total_risk_grid = ma.masked_invalid(total_risk_grid)

                # Calculate CVaR for each cell in the risk grid
                cvar_combined_risk = compute_cvar_cellwise(masked_total_risk_grid)

                # Define a custom colormap (gray to yellow to red)
                colors = [(0.5, 0.5, 0.5), (1, 1, 0), (1, 0, 0)]
                cmap = LinearSegmentedColormap.from_list("gray_yellow_red", colors)

                ax.clear()
                # Note: pcolormesh expects data in shape (ny, nx), so we transpose the risk grid for plotting.
                c = ax.pcolormesh(X, Y, cvar_combined_risk.T, shading='auto', cmap=cmap, alpha=0.7)
                if colorbar is None:
                    colorbar = fig.colorbar(c, ax=ax, label='Risk Value (0 = safe, 1 = risky)')
                else:
                    colorbar.update_normal(c)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title('Risk Visualization')

                ###############################################
                # A* Path Planning Integration
                ###############################################
                # Get the current vehicle position (world coordinates)
                current_position, _ = lidar_test.get_vehicle_pose()
                start_world = current_position[:2]  # Use only x and y
                
                # Convert start and final goal from world coordinates to grid indices.
                start_idx = world_to_grid(start_world[0], start_world[1], x_edges, y_edges)
                goal_idx = world_to_grid(final_goal_world[0], final_goal_world[1], x_edges, y_edges)
                
                # Run A* on the risk grid.
                # Note: cvar_combined_risk is of shape (len(x_mid), len(y_mid)) where
                # first dimension corresponds to x and second to y.
                path = a_star_planning(cvar_combined_risk, start_idx, goal_idx, grid_resolution)
                
                if path is not None:
                    # Convert grid indices back to world coordinates for plotting.
                    path_world = []
                    for (i, j) in path:
                        # Using the grid edges, find the center of the cell.
                        x_world = x_edges[i] + grid_resolution / 2
                        y_world = y_edges[j] + grid_resolution / 2
                        path_world.append((x_world, y_world))
                    path_world = np.array(path_world)
                    
                    # Plot the path on top of the risk map.
                    ax.plot(path_world[:, 0], path_world[:, 1], color='cyan', linewidth=2, label='Planned Path')
                    ax.legend()
                else:
                    print("No path found from start to goal.")

                plt.draw()
                plt.pause(0.1)
    finally:
        plt.ioff()
        plt.show()
