# import setup_path
import cosysairsim as airsim
import numpy as np
import open3d as o3d
import time
from linefit import ground_seg
import os
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
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
        # print(f"Position: {position_array}")
        # Convert quaternion orientation to a rotation matrix
        q = orientation
        rotation_matrix = self.quaternion_to_rotation_matrix(q)

        return position_array, rotation_matrix

    def quaternion_to_rotation_matrix(self, q):
        # Convert quaternion to a 3x3 rotation matrix
        q0, q1, q2, q3 = q.w_val, q.x_val, -q.y_val, q.z_val
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
                                
        return rot_matrix

    def transform_to_world(self, points, position, rotation_matrix):
        # Apply rotation and translation to transform points to the world coordinate frame
        points_in_world = np.dot(points+position, rotation_matrix.T) 
        return points_in_world

class GridMap:
    def __init__(self, resolution):
        self.resolution = resolution
        self.grid = {}

    def get_grid_cell(self, x, y):
        # Convert x, y coordinates to grid cell indices
         
        grid_x = round(int(x / self.resolution),1)
        grid_y = round(int(y / self.resolution),1)

        return grid_x, grid_y

    def add_point(self, x, y, z, timestamp):
        # Get grid cell for the point
        cell = self.get_grid_cell(x, y)
        
        if cell not in self.grid:
            self.grid[cell] = []
        
        # Store point with rgb, intensity, and timestamp
        self.grid[cell].append((x, y, z, timestamp))

    def sort_cells_by_time(self):
        # Sort points in each cell by timestamp
        for cell in self.grid:
            self.grid[cell].sort(key=lambda p: p[3])  # Sort by timestamp (6th element)
    
    def get_average_point_per_cell(self):
        # Return the average point (x, y, z) for each cell
        averaged_points = []

        for cell, points in self.grid.items():
            # Average x, y, z
            avg_point = np.mean([p[:3] for p in points], axis=0)
            averaged_points.append(avg_point)
        return np.array(averaged_points)
# Main
if __name__ == "__main__":
    # Initialize Lidar test
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    grid_map = GridMap(resolution=0.01)
    obstacle = GridMap(resolution=0.01)

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
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.ion()  # Enable interactive mode
    colorbar = None
    try:
        while True:
            point_cloud_data, timestamp = lidar_test.get_data(gpulidar=True)
            if point_cloud_data is not None:
                points = np.array(point_cloud_data[:, :3], dtype=np.float64)
                points = points[np.linalg.norm(points, axis=1) > 0.6]  # Filter points

                points[:, 2] = -points[:, 2]  # Reverse Z-axis if necessary

                # Get the vehicle's pose in world coordinates
                position, rotation_matrix = lidar_test.get_vehicle_pose()

                # Transform points to world coordinates
                points_world = lidar_test.transform_to_world(points, position, rotation_matrix)

                # Run ground segmentation (label == 1 for ground, label == 0 for obstacles)
                label = np.array(groundseg.run(points_world))

                # Add to grid map (ground and obstacles separately)
                for i, point in enumerate(points_world[label == 1]):  # Ground points
                    x, y, z = point
                    grid_map.add_point(x, y, z, timestamp)
                
                for i, point in enumerate(points_world[label == 0]):  # Obstacle points
                    x, y, z = point
                    obstacle.add_point(x, y, z, timestamp)

                # Sort and average points per grid cell
                grid_map.sort_cells_by_time()
                averaged_points = grid_map.get_average_point_per_cell()

                obstacle.sort_cells_by_time()
                averaged_obstacle = obstacle.get_average_point_per_cell()

                grid_point_cloud = o3d.geometry.PointCloud()
                grid_point_cloud.points = o3d.utility.Vector3dVector(averaged_points)
                # grid_point_cloud.colors = o3d.utility.Vector3dVector(averaged_colors)

                #save the point cloud
                o3d.io.write_point_cloud("grid_point_cloud.ply", grid_point_cloud)


                # Convert grid map to numpy arrays
                points = np.asarray(averaged_points)
                obstacle_points = np.asarray(averaged_obstacle)

                # Extract X, Y, Z coordinates
                x = points[:, 0]
                y = points[:, 1]
                z = points[:, 2]

                x_obstacle = obstacle_points[:, 0]
                y_obstacle = obstacle_points[:, 1]
                z_obstacle = obstacle_points[:, 2]

                # Define the number of bins (grid resolution)
                num_bins = 200

                # Compute bin edges based on the combined ground and obstacle points
                combined_x = np.concatenate((x, x_obstacle))
                combined_y = np.concatenate((y, y_obstacle))

                # Compute the bin edges once using the combined X and Y data
                x_edges = np.linspace(-10, 12, num_bins + 1)
                y_edges = np.linspace(-10, 10, num_bins + 1)

                # Compute the 2D histograms for ground and obstacle points using the same bin edges
                stat, _, _, _ = binned_statistic_2d(x, y, z, statistic='mean', bins=[x_edges, y_edges])
                stat_obstacle, _, _, _ = binned_statistic_2d(x_obstacle, y_obstacle, z_obstacle, statistic='mean', bins=[x_edges, y_edges])

                # Create a meshgrid for visualization
                x_mid = (x_edges[:-1] + x_edges[1:]) / 2
                y_mid = (y_edges[:-1] + y_edges[1:]) / 2
                X, Y = np.meshgrid(x_mid, y_mid)

                # Clear previous plot
                ax.clear()

                # Plot the ground points data
                pcolormesh = ax.pcolormesh(X, Y, stat.T, cmap='terrain')

                # Overlay obstacle cells in red
                pcolormesh_obstacle = ax.pcolormesh(X, Y, stat_obstacle.T, cmap='Reds', alpha=0.6)

                # Update or create colorbar only once
                if colorbar is None:
                    colorbar = fig.colorbar(pcolormesh, ax=ax, label='Average Elevation (Z)')
                else:
                    pcolormesh.set_clim(vmin=np.nanmin(stat), vmax=np.nanmax(stat))
                    colorbar.update_normal(pcolormesh)

                ax.set_title("Binned 2.5D Elevation Map with Obstacles (Red)")
                ax.set_xlabel("Y")
                ax.set_ylabel("X")

                # Redraw the plot
                fig.canvas.draw()
                plt.pause(0.001)

    finally:
        print("Visualization ended.")

        # cubic tranorm