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
        w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
        rot_matrix = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                               [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                               [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])
        return rot_matrix

    def transform_to_world(self, points, position, rotation_matrix):
        # Apply rotation and translation to transform points to the world coordinate frame
        points_in_world = np.dot(points, rotation_matrix.T) + position
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
                # Extract x, y, z, rgb, and intensity
                points = np.array(point_cloud_data[:, :3], dtype=np.float64)
                points = points[np.linalg.norm(points, axis=1) > 0.6]

                points[:, 2] = -points[:, 2]

                # Get the vehicle's pose in world coordinates
                position, rotation_matrix = lidar_test.get_vehicle_pose()

                # Transform points to world coordinates
                points_world = lidar_test.transform_to_world(points, position, rotation_matrix)

                # Run ground segmentation
                label = np.array(groundseg.run(points_world))

                # Add to grid map where points have label 1 (ground)
                for i, point in enumerate(points_world[label == 1]):
                    x, y, z = point
                    grid_map.add_point(x, y, z, timestamp)

                # Sort points in each cell by time stamp
                grid_map.sort_cells_by_time()
                averaged_points = grid_map.get_average_point_per_cell()

                # # Ensure correct dimensionality
                # if len(averaged_points) > 0 and averaged_points.shape[1] == 3:
                #     grid_point_cloud = o3d.geometry.PointCloud()
                #     grid_point_cloud.points = o3d.utility.Vector3dVector(averaged_points)
                #     grid_point_cloud.colors = o3d.utility.Vector3dVector(averaged_colors)
                # Create Open3D point cloud from the grid
                grid_point_cloud = o3d.geometry.PointCloud()
                grid_point_cloud.points = o3d.utility.Vector3dVector(averaged_points)

                # Convert the point cloud to a numpy array
                points = np.asarray(grid_point_cloud.points)

                # Extract X, Y, Z coordinates
                x = points[:, 0]
                y = points[:, 1]
                z = points[:, 2]

                # Define the number of bins (grid resolution)
                num_bins = 100

                # Compute the 2D histogram to bin the points and calculate the mean of Z for each bin
                stat, x_edges, y_edges, bin_numbers = binned_statistic_2d(x, y, z, statistic='mean', bins=num_bins)

                stat = np.fliplr(stat)
                stat = np.rot90(stat)
                
                # Create a meshgrid for visualization
                x_mid = (x_edges[:-1] + x_edges[1:]) / 2
                y_mid = (y_edges[:-1] + y_edges[1:]) / 2
                X, Y = np.meshgrid(y_mid, x_mid)

                # Clear previous plot
                ax.clear()

                # Plot the updated data
                pcolormesh = ax.pcolormesh(X, Y, stat.T, cmap='terrain')

                # Update or create colorbar only once
                if colorbar is None:
                    colorbar = fig.colorbar(pcolormesh, ax=ax, label='Average Elevation (Z)')
                else:
                    pcolormesh.set_clim(vmin=np.nanmin(stat), vmax=np.nanmax(stat))
                    colorbar.update_normal(pcolormesh)

                ax.set_title("Binned 2.5D Elevation Map from Point Cloud (Averaged per Grid)")
                ax.set_xlabel("Y")
                ax.set_ylabel("X")



                # Redraw the plot
                fig.canvas.draw()
                plt.pause(0.001)

    finally:
        i =0 
        # vis.destroy_window()