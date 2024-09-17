# import setup_path
import cosysairsim as airsim
import numpy as np
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
        grid_x = float(x // self.resolution)
        grid_y = float(y // self.resolution)
        return grid_x, grid_y

    def add_point(self, x, y, z, rgb, intensity, timestamp):
        # Get grid cell for the point
        cell = self.get_grid_cell(x, y)
        
        if cell not in self.grid:
            self.grid[cell] = []
        
        # Store point with rgb, intensity, and timestamp
        self.grid[cell].append((x, y, z, rgb, intensity, timestamp))

    def sort_cells_by_time(self):
        # Sort points in each cell by timestamp
        for cell in self.grid:
            self.grid[cell].sort(key=lambda p: p[5])  # Sort by timestamp (6th element)
    
    def get_average_point_per_cell(self):
        # Return the average point (x, y, z) for each cell
        averaged_points = []
        colors = []
        for cell, points in self.grid.items():
            # Average x, y, z
            avg_point = np.mean([p[:3] for p in points], axis=0)
            avg_rgb = np.mean([p[3] for p in points], axis=0)
            averaged_points.append(avg_point)
            colors.append(avg_rgb / 255.0)  # Normalize to [0, 1] for RGB
        return np.array(averaged_points), np.array(colors)
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
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Lidar Visualization', width=800, height=600)

    try:
        while True:
            point_cloud_data, timestamp = lidar_test.get_data(gpulidar=True)
            if point_cloud_data is not None:
                # Extract x, y, z, rgb, and intensity
                points = np.array(point_cloud_data[:, :3], dtype=np.float64)
                rgb_values = point_cloud_data[:, 3].astype(np.uint32)  # Extract RGB from float32 representation
                intensity = point_cloud_data[:, 4]  # Intensity values

                # Convert RGB values from float32 to RGB8
                rgb = np.zeros((np.shape(points)[0], 3))
                for index, rgb_value in enumerate(rgb_values):
                    rgb[index, 0] = (rgb_value >> 16) & 0xFF  # Extract red channel
                    rgb[index, 1] = (rgb_value >> 8) & 0xFF   # Extract green channel
                    rgb[index, 2] = rgb_value & 0xFF          # Extract blue channel

                # Reverse z-axis in local coordinates
                points[:, 2] = -points[:, 2]

                # Get the vehicle's pose in world coordinates
                position, rotation_matrix = lidar_test.get_vehicle_pose()

                # Transform points to world coordinates
                points_world = lidar_test.transform_to_world(points, position, rotation_matrix)

                
                # # Compute the center point of the point cloud
                # center_point = np.mean(points_world, axis=0)

                # # Create a point cloud for the center point with green color
                # center_point_cloud = o3d.geometry.PointCloud()
                # center_point_cloud.points = o3d.utility.Vector3dVector([center_point])

                # # Set the color of the center point to green ([0, 1, 0] in RGB)
                # center_point_cloud.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

                # Run ground segmentation
                label = np.array(groundseg.run(points_world))

                # Add to grid map where points have label 1 (ground)
                for i, point in enumerate(points_world[label == 1]):
                    x, y, z = point
                    rgb_val = rgb[i]
                    intensity_val = intensity[i]
                    grid_map.add_point(x, y, z, rgb_val, intensity_val, timestamp)

                print(f"Number of cells: {len(grid_map.grid)}")
                # Sort points in each cell by time stamp
                grid_map.sort_cells_by_time()

                # Visualize the grid map
                averaged_points, averaged_colors = grid_map.get_average_point_per_cell()

                # Check the shape of the averaged points and colors
                print(f"Shape of averaged_points: {averaged_points.shape}")
                print(f"Shape of averaged_colors: {averaged_colors.shape}")

                # Ensure correct dimensionality
                if len(averaged_points) > 0 and averaged_points.shape[1] == 3:
                    grid_point_cloud = o3d.geometry.PointCloud()
                    grid_point_cloud.points = o3d.utility.Vector3dVector(averaged_points)
                    grid_point_cloud.colors = o3d.utility.Vector3dVector(averaged_colors)

                # Create Open3D point cloud from the grid
                grid_point_cloud = o3d.geometry.PointCloud()
                grid_point_cloud.points = o3d.utility.Vector3dVector(averaged_points)
                grid_point_cloud.colors = o3d.utility.Vector3dVector(averaged_colors)

                voxel_size = 0.2  # Adjust as necessary
                grid_point_cloud = grid_point_cloud.voxel_down_sample(voxel_size=voxel_size)
                # Clear previous geometries
                vis.clear_geometries()

                # Add the new grid point cloud
                vis.add_geometry(grid_point_cloud)

                # Update visualization
                vis.poll_events()
                vis.update_renderer()

            time.sleep(0.1)
    finally:
        vis.destroy_window()