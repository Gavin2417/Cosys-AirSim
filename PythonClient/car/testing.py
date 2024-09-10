# import setup_path
import cosysairsim as airsim
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import time
from linefit import ground_seg
import os

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
        print(f"lidarData: {lidarData.time_stamp}")
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
                return points
        else:
            return None
class GridMap:
    def __init__(self, resolution):
        self.resolution = resolution
        self.grid = {}

    def get_grid_cell(self, x, y):
        # Convert x, y coordinates to grid cell indices
        grid_x = int(x // self.resolution)
        grid_y = int(y // self.resolution)
        return grid_x, grid_y

    def add_point(self, x, y, z, timestamp):
        # Get grid cell for the point
        cell = self.get_grid_cell(x, y)
        
        if cell not in self.grid:
            self.grid[cell] = []
        
        # Store point with timestamp
        self.grid[cell].append((x, y, z, timestamp))

    def sort_cells_by_time(self):
        # Sort points in each cell by timestamp
        for cell in self.grid:
            self.grid[cell].sort(key=lambda p: p[3])  # Sort by timestamp (4th element)
    
    def get_cell_points(self, cell):
        return self.grid.get(cell, [])
# Main
if __name__ == "__main__":
    # Initialize Lidar test
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    grid_map = GridMap(resolution=1.0)
    # # Set speed to 5 m/s
    # lidar_test.client.enableApiControl(True, 'CPHusky')
    # lidar_test.client.setCarControls(airsim.CarControls(throttle=0.5, steering=0.0))

    # Initialize ground segmentation object with default or config file
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
    config_path = f"{BASE_DIR}/assets/config.toml"
    
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, using default parameters")
        groundseg = ground_seg()
    else:
        groundseg = ground_seg(config_path)

    # Initialize visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    try:
        while True:
            point_cloud_data = lidar_test.get_data(gpulidar=True)
            # print(f"Point cloud data: {point_cloud_data}")
            if point_cloud_data is not None:
                # Extract only x, y, z coordinates
                points = np.array(point_cloud_data[:, :3], dtype=np.float64)
                points[:,2] = -points[:,2]  # Reverse z-axis
                # Run ground segmentation
                label = np.array(groundseg.run(points))

                # Create point cloud for all points
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(points)

                # Color points based on ground/obstacle classification
                colors = np.zeros((points.shape[0], 3))
                colors[label == 1] = [0, 1, 0]  # Green for ground points
                colors[label == 0] = [1, 0, 0]  # red for obstacle points
                point_cloud.colors = o3d.utility.Vector3dVector(colors)

                #get the grounds point as a numpy array
                ground_points = points[label == 1]
                # print(f"Ground points: {len(ground_points)}")  

                # Clear previous geometries
                vis.clear_geometries()

                # Add the new point cloud
                vis.add_geometry(point_cloud)

                # Update visualization
                vis.poll_events()
                vis.update_renderer()

            time.sleep(0.1)
    finally:
        vis.destroy_window()
