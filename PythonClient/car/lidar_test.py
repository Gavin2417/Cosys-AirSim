import setup_path
import cosysairsim as airsim
import numpy as np
import matplotlib.pyplot as plt

class lidarTest:

    def __init__(self, lidar_name, vehicle_name):

        # connect to the AirSim simulator
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.vehicleName = vehicle_name
        self.lidarName = lidar_name
        self.lastlidarTimeStamp = 0

    def get_data(self, gpulidar):

        # get lidar data
        if gpulidar:
            lidarData = self.client.getGPULidarData(self.lidarName, self.vehicleName)
          
            if lidarData.time_stamp != self.lastlidarTimeStamp:
                # Check if there are any points in the data
                if len(lidarData.point_cloud) < 5:
                    self.lastlidarTimeStamp = lidarData.time_stamp
                    return None
                else:
                    self.lastlidarTimeStamp = lidarData.time_stamp

                    points = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
                    points = np.reshape(points, (int(points.shape[0] / 5), 5))
                    
                    return points
            else:
                return None
        else:
            lidarData = self.client.getLidarData(self.lidarName, self.vehicleName)

            if lidarData.time_stamp != self.lastlidarTimeStamp:
                # Check if there are any points in the data
                if len(lidarData.point_cloud) < 5:
                    self.lastlidarTimeStamp = lidarData.time_stamp
                    return None
                else:
                    self.lastlidarTimeStamp = lidarData.time_stamp

                    points = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
                    points = np.reshape(points, (int(points.shape[0] / 3), 3))
                    points = points * np.array([1, -1, -1])
                    
                    return points
            else:
                return None

    def stop(self):

        self.client.reset()
        print("Stopped!\n")


# main
if __name__ == "__main__":

    lidarTest = lidarTest('gpulidar1', 'CPHusky')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    gpulidar = True
    while True:
        # move forward
        lidarTest.client.enableApiControl(True, 'CPHusky')
        lidarTest.client.setCarControls(airsim.CarControls(throttle=0.5, steering=0.5), 'CPHusky')

        points = lidarTest.get_data(gpulidar)

        # Check if points is None before proceeding
        if points is not None:
            print(points)
            print(f"Number of points: {len(points)}")
            
            # You can add code here to plot or process the points
            ax.scatter(points[:, 0], points[:, 1], points[:, 2])
            plt.pause(0.001)
            ax.cla()
        else:
            print("No points")
