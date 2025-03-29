# import setup_path
import cosysairsim as airsim
import numpy as np
import open3d as o3d
import time
from linefit import ground_seg
import os
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
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
class KalmanFilter:
    def __init__(self, process_noise=1e-4, measurement_noise=0.1, initial_estimate=0, initial_uncertainty=1):
        self.z = initial_estimate  # Estimated height
        self.P = initial_uncertainty  # Estimated uncertainty
        self.Q = process_noise  # Process noise
        self.R = measurement_noise  # Measurement noise

    def predict(self):
        # In a simple case, we do not have a control input, so the prediction is just the prior estimate.
        self.P += self.Q  # Increase uncertainty with the process noise (prediction step)

    def update(self, z_measure):
        # Kalman gain
        K = self.P / (self.P + self.R)
        
        # Update the estimate with measurement
        self.z = self.z + K * (z_measure - self.z)
        
        # Update the uncertainty
        self.P = (1 - K) * self.P
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
            # Initialize a Kalman filter for each new grid cell
            self.grid[cell] = KalmanFilter(initial_estimate=z)

        # Get the Kalman filter for this cell
        kf = self.grid[cell]

        # Kalman Filter Prediction Step
        kf.predict()

        # Kalman Filter Update Step with the new measurement (z)
        kf.update(z)

        # Store the updated Kalman filter in the grid
        self.grid[cell] = kf

    def get_height_estimate_and_uncertainty(self):
        # Return the estimated height and uncertainty for each cell
        estimated_points = []
        uncertainties = []
        
        for cell, kf in self.grid.items():
            # Get the grid cell center position from the cell index
            x, y = cell
            # Store the estimated height and uncertainty
            estimated_points.append([x * self.resolution, y * self.resolution, kf.z])
            uncertainties.append(kf.P)  # Uncertainty for visualization

        return np.array(estimated_points)
# Main
if __name__ == "__main__":
    # Initialize Lidar test
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    grid_map = GridMap(resolution=0.05)

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
    # Initialize 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colorbar = None
    try:
        while True:
            point_cloud_data, timestamp = lidar_test.get_data(gpulidar=True)
            if point_cloud_data is not None:
                # Extract x, y, z coordinates from point cloud
                points = np.array(point_cloud_data[:, :3], dtype=np.float64)
                points = points[np.linalg.norm(points, axis=1) > 0.6]
                

                # Get the vehicle's pose in world coordinates
                position, rotation_matrix = lidar_test.get_vehicle_pose()

                # Transform points to world coordinates
                points_world = lidar_test.transform_to_world(points, position, rotation_matrix)
                points_world[:, 2] = -points_world[:, 2]  # Flip Z-axis if needed
                label = np.array(groundseg.run(points_world))
                
                # Add points to the grid map
                for i, point in enumerate(points_world[label == 1]):
                    x, y, z = point
                    grid_map.add_point(x, y, z, timestamp)
                
                # Get the average point per cell
                averaged_points = grid_map.get_height_estimate_and_uncertainty()
                
                
                # Extract the x, y, z values from averaged_points
                x_vals = averaged_points[:, 0]
                y_vals = averaged_points[:, 1]
                z_vals = averaged_points[:, 2]

                ### Option 1: Apply Offset to Ensure Positive Z values
                offset = abs(min(z_vals))  # This ensures that the minimum value is shifted to 0
                z_vals_offset = z_vals + offset 
                
                # Define width and depth for the bars
                width = depth = 0.05  # Based on grid resolution

                # Clear previous plot
                ax.clear()

                # Plot the bars using bar3d
                ax.bar3d(x_vals, y_vals, np.zeros_like(z_vals), width, depth, z_vals_offset, shade=True)

                # Set axis labels and title
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('3D Bar Elevation Map')

                # Redraw the plot
                plt.draw()
                plt.pause(0.1)

    finally:
        plt.ioff()  # Disable interactive mode
        plt.show()