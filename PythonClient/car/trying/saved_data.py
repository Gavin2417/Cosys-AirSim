import os
import math
import time
import heapq
import numpy as np
import open3d as o3d
import cosysairsim as airsim
import cv2
from function5 import calculate_combined_risks, compute_cvar_cellwise
from scipy.ndimage import generic_filter
import json
class lidarTest:
    def __init__(self, lidar_name, vehicle_name):
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.vehicleName = vehicle_name
        self.lidarName = lidar_name
        self.lastlidarTimeStamp = 0

    def get_data(self, gpulidar):
        if gpulidar:
            lidarData = self.client.getGPULidarData(self.lidarName, self.vehicleName)
        else:
            lidarData = self.client.getLidarData(self.lidarName, self.vehicleName)
        if lidarData.time_stamp != self.lastlidarTimeStamp:
            self.lastlidarTimeStamp = lidarData.time_stamp
            if len(lidarData.point_cloud) < 2:
                return None, None
            points = np.array(lidarData.point_cloud, dtype=np.float32)
            num_dims = 5 if gpulidar else 3
            points = points.reshape((-1, num_dims))
            if not gpulidar:
                points = points * np.array([1, -1, 1])
            return points, lidarData.time_stamp
        return None, None

    def get_vehicle_pose(self):
        vehicle_pose = self.client.simGetVehiclePose()
        pos = vehicle_pose.position
        orient = vehicle_pose.orientation
        position_array = np.array([float(pos.x_val), float(pos.y_val), float(pos.z_val)])
        rotation_matrix = self.quaternion_to_rotation_matrix(orient)
        return position_array, rotation_matrix

    def quaternion_to_rotation_matrix(self, q):
        qw, qx, qy, qz = q.w_val, q.x_val, q.y_val, q.z_val
        return np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])

    def transform_to_world(self, points, position, rotation_matrix):
        points_rotated = np.dot(points, rotation_matrix.T)
        return points_rotated + position
    def is_flipped(self, threshold=-0.7):
        """Check if the vehicle is flipped upside down."""
        orient = self.client.simGetVehiclePose().orientation
        rot_matrix = self.quaternion_to_rotation_matrix(orient)
        up_vector = rot_matrix[:, 2]  # Z-axis in world frame
        return up_vector[2] < threshold 
def create_incrementing_folder(base_name="track", parent_folder="test"):
    os.makedirs(parent_folder, exist_ok=True)  # Ensure 'test' folder exists
    i = 0
    while True:
        folder_name = os.path.join(parent_folder, f"{i}")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Created folder: {folder_name}")
            return folder_name
        i += 1
def serialize(obj):
    if hasattr(obj, "__dict__"):
        return {k: serialize(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    else:
        # primitive (int, float, bool, str, etc.)
        return obj

if __name__ == "__main__":
    output_folder = "data"
    track_folder = create_incrementing_folder(base_name="track", parent_folder=output_folder)


    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    counter = 0

    try:
        while True:
            point_cloud_data, timestamp = lidar_test.get_data(gpulidar=True)
            if point_cloud_data is None:
                continue

            points = np.array(point_cloud_data[:, :3], dtype=np.float64)
            points = points[np.linalg.norm(points, axis=1) > 0.6]
            position, rotation_matrix = lidar_test.get_vehicle_pose()
            # points_world = lidar_test.transform_to_world(points, position, rotation_matrix)
            points_world = points.copy()
            points_world[:, 2] = -points_world[:, 2]

            length = len(points_world)
            # print(f"Length of point cloud: {length}")
            # Save point cloud to .ply
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world)
            ply_filename = os.path.join(track_folder, f"{counter}.ply")
            o3d.io.write_point_cloud(ply_filename, pcd)

            # Get and save camera image properly
            png_image = lidar_test.client.simGetImage("0", airsim.ImageType.Scene)
            if png_image is not None and len(png_image) > 0:
                img_filename = os.path.join(track_folder, f"{counter}.png")
                with open(img_filename, "wb") as f:
                    f.write(png_image)

            # save the car information
            car_state = lidar_test.client.getCarState()
            # print(car_state)
            car_state_filename = os.path.join(track_folder, f"{counter}_car_state.json")
            car_state_dict = serialize(car_state)
    

            with open(car_state_filename, "w") as f:
                json.dump(car_state_dict, f, indent=2)

            # Get the collision information
            collision_info = lidar_test.client.simGetCollisionInfo()
            collision_info_filename = os.path.join(track_folder, f"{counter}_collision_info.json")
            collision_info_dict = serialize(collision_info)
            with open(collision_info_filename, "w") as f:
                json.dump(collision_info_dict, f, indent=2)
            
            
            # if collision_info.has_collided:
            #     break


            counter += 1
            print(f"Saved {counter} point clouds and images to {track_folder}")
            # the husky is upside down, so we need to stop when it is upside down
            if counter >= 4000 or lidar_test.is_flipped():
                print("Reached 4000 point clouds, stopping.")
                break

    except KeyboardInterrupt:
        pass
