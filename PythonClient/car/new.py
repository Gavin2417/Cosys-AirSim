# import setup_path
import cosysairsim as airsim
import numpy as np
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
            if len(lidarData.point_cloud) < 5:
                self.lastlidarTimeStamp = lidarData.time_stamp
                return None
            else:
                self.lastlidarTimeStamp = lidarData.time_stamp
                # Process lidar point cloud data
                points = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
                num_dims = 5 if gpulidar else 3
                points = np.reshape(points, (int(points.shape[0] / num_dims), num_dims))
                if not gpulidar:
                    points = points * np.array([1, -1, -1])  # Adjust for AirSim coordinates
                return points
        else:
            return None

    def segment_ground(self, point_cloud):
        # Simple ground segmentation based on height (e.g., points with Z < a threshold)
        ground_threshold = 0.5  # Customize threshold based on environment
        ground_points = point_cloud[point_cloud[:, 2] > ground_threshold]
        obstacle_points = point_cloud[point_cloud[:, 2] <= ground_threshold]
        return ground_points, obstacle_points

    def build_elevation_map(self, ground_points, grid_resolution=0.1):
        # Generate a 2.5D elevation map from the ground points
        x_min, y_min = np.min(ground_points[:, :2], axis=0)
        x_max, y_max = np.max(ground_points[:, :2], axis=0)
        
        x_bins = np.arange(x_min, x_max, grid_resolution)
        y_bins = np.arange(y_min, y_max, grid_resolution)
        height_map = np.zeros((len(x_bins), len(y_bins)))
        
        for point in ground_points:
            x_idx = np.searchsorted(x_bins, point[0]) - 1
            y_idx = np.searchsorted(y_bins, point[1]) - 1
            height_map[x_idx, y_idx] = max(height_map[x_idx, y_idx], point[2])

        smoothed_height_map = gaussian_filter(height_map, sigma=1)
        return smoothed_height_map

    def compute_risk_map(self, height_map):
        # Example risk map based on height differences
        risk_map = np.abs(np.gradient(height_map)[0])  # Gradient of the height map
        return risk_map

    def plot_maps(self, height_map, risk_map):
        # Visualize the height map and risk map
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Elevation Map")
        plt.imshow(height_map, cmap='terrain')
        plt.colorbar(label='Height')

        plt.subplot(1, 2, 2)
        plt.title("Risk Map")
        plt.imshow(risk_map, cmap='hot')
        plt.colorbar(label='Risk')
        plt.show()

    def stop(self):
        self.client.reset()
        print("Stopped!\n")


# main
if __name__ == "__main__":

    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    while True:
        points = lidar_test.get_data(gpulidar=True)
        if points is not None:
            ground_points, obstacle_points = lidar_test.segment_ground(points)
            # plot the ground points
            ax.clear()
            ax.scatter(ground_points[:, 0], ground_points[:, 1], ground_points[:, 2], c='g', s=1)
            ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], obstacle_points[:, 2], c='r', s=1)
            plt.pause(0.01)
    
            
