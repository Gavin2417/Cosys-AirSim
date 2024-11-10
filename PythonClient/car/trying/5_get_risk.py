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
from risk_calculator import calculate_slope_risk, calculate_step_risk, combine_risks, compute_cvar_cellwise, plot_cvar_grid
from matplotlib.colors import LinearSegmentedColormap
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

                # Classify ground and obstacle points
                for i, point in enumerate(points_world):
                    x, y, z = point
                    if label[i] == 1:
                        grid_map_ground.add_point(x, y, z, timestamp)
                    elif z > -0.2:
                        grid_map_obstacle.add_point(x, y, z, timestamp)

                ground_points = grid_map_ground.get_height_estimate()
                obstacle_points = grid_map_obstacle.get_height_estimate()

                # Calculate risks
                # Calculate step risk and retrieve X, Y grids
                step_risk_grid, X, Y = calculate_step_risk(ground_points)

                # Calculate slope risk, now aligned with step_risk_grid dimensions
                slope_risk_grid = calculate_slope_risk(ground_points, Y,X)
                
                # Define obstacle grid based on obstacle points
                obstacle_grid = np.full(step_risk_grid.shape, 0.0)
                for i in range(len(obstacle_points)):
                    x_idx = np.digitize(obstacle_points[i, 0], X[0]) - 1
                    y_idx = np.digitize(obstacle_points[i, 1], Y[:, 0]) - 1
                    if 0 <= x_idx < X.shape[0] and 0 <= y_idx < Y.shape[1]:
                        obstacle_grid[x_idx, y_idx] = 1.0

                # Combine risks
                combined_risk_grid = combine_risks(slope_risk_grid, step_risk_grid, obstacle_grid)

                # Calculate CVaR-adjusted risk
                # cvar_combined_risk = compute_cvar_cellwise(combined_risk_grid)

                # Plot CVaR-adjusted grid
                colors = [(0.5, 0.5, 0.5), (1, 1, 0), (1, 0, 0)]
                cmap = LinearSegmentedColormap.from_list("gray_yellow_red", colors)

                ax.clear()
                c = ax.pcolormesh(X, Y, combined_risk_grid.T, shading='auto', cmap='terrain', alpha=0.7)

                # Plot obstacle points using scatter for distinct visualization
                # ax.scatter(x_vals_obstacle, y_vals_obstacle, c='red', s=10, label='Obstacles')

                # Add or update the color bar
                if colorbar is None:
                    colorbar = fig.colorbar(c, ax=ax, label='Ground Z Value (Height)')
                else:
                    colorbar.update_normal(c)

                # Set axis labels and title
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title('2D Grid Cell Plot of Ground and Obstacle Points')

                plt.draw()
                plt.pause(0.1)

    finally:
        plt.ioff()  # Disable interactive mode
        plt.show()