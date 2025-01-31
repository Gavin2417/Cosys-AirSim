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
from scipy.spatial import cKDTree
import numpy as np

def interpolate_in_radius(grid, radius):
    """
    Interpolates NaN values in a grid using valid points within a specified radius.
    
    Parameters:
        grid (ndarray): 2D grid with NaN values to interpolate.
        radius (float): Radius within which to search for valid points.
    
    Returns:
        ndarray: Grid with interpolated values.
    """
    # Get valid (non-NaN) points
    valid_points = ~np.isnan(grid)
    valid_coords = np.column_stack(np.where(valid_points))
    valid_values = grid[valid_points]

    # Create KDTree for efficient neighbor search
    tree = cKDTree(valid_coords)

    # Get NaN points
    nan_coords = np.column_stack(np.where(np.isnan(grid)))

    # Iterate through each NaN point
    for idx, coord in enumerate(nan_coords):
        # Find all valid points within the radius
        neighbors = tree.query_ball_point(coord, radius)

        # If there are neighbors, compute a weighted average
        if neighbors:
            weights = []
            weighted_values = []
            for neighbor_idx in neighbors:
                neighbor_coord = valid_coords[neighbor_idx]
                value = valid_values[neighbor_idx]

                # Compute weight based on inverse distance
                distance = np.linalg.norm(coord - neighbor_coord)
                weight = 1 / (distance + 1e-6)  # Add small epsilon to avoid division by zero
                weights.append(weight)
                weighted_values.append(weight * value)

            # Interpolated value is weighted average
            grid[coord[0], coord[1]] = np.sum(weighted_values) / np.sum(weights)

    return grid
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
                
                # Interpolate missing (NaN) values in the risk grid
                interpolation_radius = 1.5  # Set the interpolation radius
                total_risk_grid = interpolate_in_radius(total_risk_grid, interpolation_radius)

                # Mask NaN values in total_risk_grid for transparency
                masked_total_risk_grid = ma.masked_invalid(total_risk_grid)

                # Calculate CVaR for each grid cell
                cvar_combined_risk = compute_cvar_cellwise(masked_total_risk_grid)

                colors = [(0.5, 0.5, 0.5), (1, 1, 0), (1, 0, 0)]
                cmap = LinearSegmentedColormap.from_list("gray_yellow_red", colors)

                ax.clear()
                c = ax.pcolormesh(X, Y, cvar_combined_risk.T, shading='auto', cmap=cmap, alpha=0.7)

                if colorbar is None:
                    colorbar = fig.colorbar(c, ax=ax, label='Risk Value (0=zero risk, 1= risky)')
                else:
                    colorbar.update_normal(c)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title('Risk Visualization')

                plt.draw()
                plt.pause(0.1)

    finally:
        plt.ioff()
        plt.show()