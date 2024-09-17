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
                return points
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


# Main
if __name__ == "__main__":
    # Initialize Lidar test
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    # grid_map = GridMap(resolution=1.0)

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
    vis.create_window(window_name='Lidar Visualization', width=800, height=800)

    try:
        while True:
            # Get the current Lidar data
            point_cloud_data = lidar_test.get_data(gpulidar=True)
            if point_cloud_data is not None:
                # Extract x, y, z coordinates
                points = np.array(point_cloud_data[:, :3], dtype=np.float64)
                points[:,2] = -points[:,2]  # Reverse z-axis to align with world coordinates

                # Get vehicle position and orientation
                vehicle_position, vehicle_rotation = lidar_test.get_vehicle_pose()

                # Transform the points to the world coordinate frame
                points_in_world = lidar_test.transform_to_world(points, vehicle_position, vehicle_rotation)

                # Run ground segmentation on the transformed points
                label = np.array(groundseg.run(points_in_world))

                # Create point cloud for all points
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(points_in_world)

                # Color points based on ground/obstacle classification
                colors = np.zeros((points_in_world.shape[0], 3))
                colors[label == 1] = [0, 1, 0]  # Green for ground points
                colors[label == 0] = [1, 0, 0]  # Red for obstacle points
                point_cloud.colors = o3d.utility.Vector3dVector(colors)

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