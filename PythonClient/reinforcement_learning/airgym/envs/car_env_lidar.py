import setup_path
import cosysairsim as airsim
import numpy as np
import math
import time
from gymnasium.spaces import Box, Dict
import gymnasium
from gymnasium import spaces
from airgym.envs.airsim_env import AirSimEnv
import os, io, json, logging
import open3d as o3d
from PIL import Image
logging.basicConfig(level=logging.INFO)

def create_incrementing_folder(parent_folder="data"):
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


class AirSimCarEnvLidar(AirSimEnv):
    def __init__(self, ip_address, image_shape):
        super().__init__(image_shape)

        self.image_shape = image_shape
        self.start_ts = 0

        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "collision": False,
        }
        
        self.car = airsim.CarClient(ip=ip_address)
        self.action_space = spaces.Discrete(4)

        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.Scene, False, True
        )

        # Add Lidar data request
        self.lidar_request = "gpulidar1"
        self.lidarTest = lidarTest('gpulidar1', 'CPHusky')
        self.observation_space = Dict({
            'image': Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
            'lidar': Box(low=-np.inf, high=np.inf, shape=(16384,5), dtype=np.float32),
            'Pose': Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        })

        self.car_controls = airsim.CarControls()
        self.car_state = None
        self.prev_dist = 0
        self.num = 0

        # Add a frame counter
        self.frame_count = 0
        # where we'll dump per‐episode recordings
        self.output_root = "../car/trying/data"
        # these get set in reset()
        self.track_folder = None  
        self.record_counter = 0

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)

        time.sleep(0.01)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action):
        # Reset car_controls
        self.car_controls.brake = 0
        self.car_controls.throttle = 0.25
        if action == 0:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
        elif action == 1:
            self.car_controls.steering = 0
        elif action == 2:
            self.car_controls.steering = 1
        else:
            self.car_controls.steering = -1

        self.car.setCarControls(self.car_controls)
        time.sleep(1)

    def transform_obs(self, response):
        # it's now a PNG
        img = Image.open(io.BytesIO(response.image_data_uint8))
        img = img.resize((84, 84), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

    def _get_obs(self):
        # Get image data
        responses = self.car.simGetImages([self.image_request])
        image = self.transform_obs(responses[0])

        # Ensure image is returned as uint8
        image = np.clip(image, 0, 255).astype(np.uint8)

        # Get Lidar data
        lidar_data = self.lidarTest.get_data(True)
        if lidar_data is None:
            # no new sweep ⇒ all zeros
            lidar_data = np.zeros((16384, 5), dtype=np.float32)
        else:
            M, D = lidar_data.shape  # D should be 5
            if M >= 16384:
                # randomly down-sample to 16384
                idx = np.random.choice(M, 16384, replace=False)
                lidar_data = lidar_data[idx]
            else:
                # pad out with zeros at the end
                pad = np.zeros((16384 - M, D), dtype=lidar_data.dtype)
                lidar_data = np.vstack([lidar_data, pad])

        # Pose data should be float32 and must remain within the bounds you set
        self.car_state = self.car.getCarState()
        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided

        pose = np.array([
            self.state["pose"].position.x_val,
            self.state["pose"].position.y_val,
            self.state["pose"].position.z_val,
            self.state["pose"].orientation.w_val,
            self.state["pose"].orientation.x_val,
            self.state["pose"].orientation.y_val,
            self.state["pose"].orientation.z_val,
        ], dtype=np.float32)

        # Return the observation in the correct format
        return {
            'image': image,
            'lidar': lidar_data,
            'Pose': pose
        }
    def compte_prev_dist(self, start_pt, goal_pt):
        self.prev_dist = np.linalg.norm(start_pt - goal_pt)
    def _compute_reward(self):
        MAX_SPEED = 300
        MIN_SPEED = 10
        THRESH_DIST = 1
        BETA = 3

        # Single goal point
        goal_pt = np.array([5.5, 5.5, 0.8])
        start_pt = np.array([-5,0, 0.8])
        initial_dist = np.linalg.norm(start_pt - goal_pt)  # Corrected typo
        
        car_pt = self.state["pose"].position.to_numpy_array()

        # Calculate Euclidean distance to the goal point
        dist = np.linalg.norm(car_pt - goal_pt)

        # Initialize self.prev_dist if this is the first step
        if self.prev_dist is None:
            self.prev_dist = initial_dist
        # Exponential reward based on distance reduction
        distance_reward = 10 * np.exp(-(dist / initial_dist)) * (self.prev_dist - dist)
        # Update previous distance
        self.prev_dist = dist

        # Terminal conditions
        done = False
        truncated = False
        
        # If the goal is reached
        if dist <= THRESH_DIST:
            distance_reward = 20  # Reward for reaching the goal
            done = True  # Mark episode as done
            self.prev_dist = None  # Reset for next episode
            truncated = False
        
        # Handling braking and speed condition
        elif self.car_controls.brake == 0 and self.car_state.speed <= 0.5:
            distance_reward = -10  # Penalty for stalling
            done = False
            truncated = True  # Mark the episode as truncated
            self.prev_dist = None  # Reset for next episode
        
        # Handling collision
        elif self.state["collision"]:
            distance_reward = -10  # Penalty for collision
            
        return distance_reward, done, truncated

    def step(self, action):
        # apply action, get obs + reward
        self._do_action(action)
        obs = self._get_obs()
        reward, done, truncated = self._compute_reward()

        # record this timestep
        folder = self.track_folder
        idx    = self.record_counter

        # -- 1) save LIDAR to PLY
        pc = obs["lidar"]  # already padded/truncated to (16384,5)
        xyz = pc[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        o3d.io.write_point_cloud(
            os.path.join(folder, f"{idx}.ply"), pcd
        )

        # -- 2) save image (grayscale uint8 H×W×1)
        img = obs["image"].squeeze()  # H×W
        im  = Image.fromarray(img)
        im.save(os.path.join(folder, f"{idx}.png"))

        # -- 3) save car state & collision info
        car_state      = self.car.getCarState()
        car_state_filename = os.path.join(folder, f"{idx}_car_state.json")
        car_state_dict = serialize(car_state)

        with open(car_state_filename, "w") as f:
            json.dump(car_state_dict, f, indent=2)

        collision_info = self.car.simGetCollisionInfo()
        collision_info_filename = os.path.join(folder, f"{idx}_collision_info.json")
        collision_info_dict = serialize(collision_info)
        with open(collision_info_filename, "w") as f:
            json.dump(collision_info_dict, f, indent=2)

        self.record_counter += 1

        # frame‐limit truncation
        self.frame_count += 1
        if self.frame_count >= 200:
            done = False
            truncated = True
            self.frame_count = 0

        return obs, reward, done, truncated, self.state


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # -- new folder for this episode
        self.track_folder   = create_incrementing_folder(self.output_root)
        self.record_counter = 0

        # reset frame counter, car, etc.
        self.frame_count = 0
        self._setup_car()
        self._do_action(1)
        self.compte_prev_dist(
            np.array([0, 0, 0.8]), np.array([5.5, 5.5, 0.8])
        )
        return self._get_obs(), {}