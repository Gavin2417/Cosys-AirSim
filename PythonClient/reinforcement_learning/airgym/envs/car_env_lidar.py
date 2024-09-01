import setup_path
import cosysairsim as airsim
import numpy as np
import math
import time
from gymnasium.spaces import Box, Dict
import gymnasium
from gymnasium import spaces
from airgym.envs.airsim_env import AirSimEnv
import logging
from PIL import Image
logging.basicConfig(level=logging.INFO)
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
            "0", airsim.ImageType.DepthPerspective, True, False
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
        img1d = np.array(response.image_data_float, dtype=np.float64)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (response.height, response.width))

        image = Image.fromarray(img2d)
        im_final = image.resize((84, 84)).convert("L")

        # save image into a file
        self.num += 1
        im_final.save(f"C:/Users/gavin/OneDrive/Documents/AirSim/test/image_{self.num}.png")

        return np.array(im_final).reshape([84, 84, 1])


    def _get_obs(self):
        # Get image data
        responses = self.car.simGetImages([self.image_request])
        image = self.transform_obs(responses[0])

        # Ensure image is returned as uint8
        image = np.clip(image, 0, 255).astype(np.uint8)

        # Get Lidar data
        lidar_data = self.lidarTest.get_data(True)
        if lidar_data is None:
            # Fill lidar_data with zeros if no data is returned
            lidar_data = np.zeros((16384, 5), dtype=np.float32)

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
            done = False
            truncated = True  # Mark episode as truncated
            self.prev_dist = None  # Reset for next episode
        return distance_reward, done, truncated

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()

        reward, done, truncated = self._compute_reward()

        # Increment the frame counter
        self.frame_count += 1

        # Check if the frame count exceeds 50
        if self.frame_count >= 30:
            done = False
            truncated = True
            self.frame_count = 0

        return obs, reward, done, truncated, self.state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Reset the frame counter
        self.frame_count = 0
        
        self._setup_car()
        self._do_action(1)
        self.compte_prev_dist(np.array([0, 0, 0.8]), np.array([5.5, 5.5, 0.8]))
        return self._get_obs(), {}
