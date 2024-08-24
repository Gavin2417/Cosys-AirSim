import setup_path
import cosysairsim as airsim
import numpy as np
import math
import time

import gymnasium
from gymnasium import spaces
from airgym.envs.airsim_env import AirSimEnv
import logging
from PIL import Image
logging.basicConfig(level=logging.INFO)
class AirSimCarEnv(AirSimEnv):
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
        # self.num += 1
        # im_final.save(f"C:/Users/gavin/OneDrive/Documents/AirSim/test/image_{self.num}.png")

        return np.array(im_final).reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.car.simGetImages([self.image_request])
        image = self.transform_obs(responses[0])

        self.car_state = self.car.getCarState()

        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided

        return image

    def _compute_reward(self):
        MAX_SPEED = 300
        MIN_SPEED = 10
        THRESH_DIST = 1
        BETA = 3

        # Single goal point
        goal_pt = np.array([4.5, 4.5, 0])
        start_pt = np.array([0, 0, 0])
        intial_dist = np.linalg.norm(start_pt - goal_pt)
        
        car_pt = self.state["pose"].position.to_numpy_array()

        # Calculate Euclidean distance to the goal point
        dist = np.linalg.norm(car_pt - goal_pt)
        distance_reward = 0
        if dist > intial_dist:
            distance_reward = -0.5
        else:
            if dist < self.prev_dist:
                distance_reward = 0.2
            else:
                distance_reward = -0.2
        self.prev_dist = dist

        done = False
        truncated = False
       
        
        if dist <= THRESH_DIST:
            distance_reward = 1
            done = True
            self.prev_dist = 9999999
            truncated = False
        else:
            if self.car_controls.brake == 0 and self.car_state.speed <= 0.5:
                distance_reward = -1
                done = False
                truncated = True
                self.prev_dist = 9999999
            if self.state["collision"]:
                distance_reward = -1
                done = False
                self.prev_dist = 9999999
                truncated = True
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
        return self._get_obs(), {}
