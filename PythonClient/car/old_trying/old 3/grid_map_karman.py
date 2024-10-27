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

        return np.array(estimated_points), np.array(uncertainties)

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

                # Add points to the grid map and update with Kalman filter
                for i, point in enumerate(points_world[label == 1]):  # Ground points
                    x, y, z = point
                    grid_map.add_point(x, y, z, timestamp)

                # Get the estimated height and uncertainties from the Kalman filter
                estimated_points, uncertainties = grid_map.get_height_estimate_and_uncertainty()

                # Convert grid map to numpy arrays for plotting
                x = estimated_points[:, 0]
                y = estimated_points[:, 1]
                z = estimated_points[:, 2]

                # Define the number of bins (grid resolution)
                num_bins = 200

                # Compute bin edges based on X and Y data
                x_edges = np.linspace(x.min(), x.max(), num_bins + 1)
                y_edges = np.linspace(y.min(), y.max(), num_bins + 1)

                # Compute the 2D histograms for ground points using the same bin edges
                stat, _, _, _ = binned_statistic_2d(x, y, z, statistic='mean', bins=[x_edges, y_edges])

                # Create a meshgrid for visualization
                x_mid = (x_edges[:-1] + x_edges[1:]) / 2
                y_mid = (y_edges[:-1] + y_edges[1:]) / 2
                X, Y = np.meshgrid(y_mid, x_mid)

                # Initialize an empty uncertainty grid with NaN values
                uncertainty_grid = np.full(X.shape, np.nan)

                # Populate the uncertainty grid with available uncertainties
                for i in range(len(estimated_points)):
                    cell_x, cell_y = grid_map.get_grid_cell(x[i], y[i])
                    # Find the bin indices corresponding to the cell_x and cell_y
                    bin_x = np.digitize(x[i], x_edges) - 1  # Get the bin index in the x direction
                    bin_y = np.digitize(y[i], y_edges) - 1  # Get the bin index in the y direction
                    
                    if 0 <= bin_x < X.shape[1] and 0 <= bin_y < X.shape[0]:  # Ensure bin index is valid
                        uncertainty_grid[bin_x, bin_x] = uncertainties[i]

                # Clear previous plot
                ax.clear()

                # Plot the ground points data
                pcolormesh = ax.pcolormesh(X, Y, stat.T, cmap='terrain')

                # Add uncertainty overlay (optional, for visualization of uncertainties)
                # uncertainty_map = ax.pcolormesh(X, Y, uncertainty_grid.T, cmap='coolwarm', alpha=0.5)

                # Update or create colorbar only once
                if colorbar is None:
                    colorbar = fig.colorbar(pcolormesh, ax=ax, label='Average Elevation (Z)')
                else:
                    pcolormesh.set_clim(vmin=np.nanmin(stat), vmax=np.nanmax(stat))
                    colorbar.update_normal(pcolormesh)

                ax.set_title("Binned 2.5D Elevation Map with Kalman Filter")
                ax.set_xlabel("Y")
                ax.set_ylabel("X")

                # Redraw the plot
                fig.canvas.draw()
                plt.pause(0.001)

    finally:
        print("Visualization ended.")