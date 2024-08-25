import setup_path
import cosysairsim as airsim
import numpy as np
import os
import time
import tempfile
import random

# Function to check for obstacles using collision detection
def GetToNew(client):
    # Save current pose
    current_pose = client.simGetVehiclePose()
    
    while True:
        # Move to test position
        new_position = airsim.Vector3r(0,0,current_pose.position.z_val)
        print("Current position: ", current_pose.position)
        print("New position: ", new_position)

        orientation = airsim.Quaternionr(0, 0, 0)
        new_pose = airsim.Pose(new_position, orientation)
        client.simSetVehiclePose(new_pose, ignore_collision=True)
        
        # Allow some time for potential collision detection
        time.sleep(1)
        
        # Check for collision
        collision_info = client.simGetCollisionInfo()
        
        # If no collision, break the loop

        if not collision_info.has_collided:
            break
        else:
            # Move back to original position
            client.simSetVehiclePose(current_pose, ignore_collision=True)
prev_dist = 999999  # This keeps track of the previous distance

def compute_exponential_reward(current_pose, goal_pt, client):
    global prev_dist

    # Calculate the current distance from the goal
    dist = np.linalg.norm(current_pose - goal_pt)
    
    # Calculate the initial distance from the start point to the goal
    start_pt = np.array([0, 0, 0.8])
    initial_dist = np.linalg.norm(start_pt - goal_pt)
    
    # Exponential reward based on distance
    distance_reward = 10*(prev_dist - dist)*np.exp(-(dist/initial_dist))
    
    prev_dist = dist

    # Check for collision
    collision_info = client.simGetCollisionInfo()
    if collision_info.has_collided:
        # Penalize heavily for collision
        distance_reward =-10
        done = True
        truncated = True
    elif dist <= 1:
        # Reward for reaching the goal
        distance_reward =20
        done = True
        truncated = False
    else:
        done = False
        truncated = False
    if distance_reward != 0:
        print(f" Reward: {distance_reward}")
    return distance_reward, done, truncated

# Connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
# Main simulation loop
while True:
    # Get the current car position
    current_pose = client.getCarState().kinematics_estimated.position.to_numpy_array()
    
    # Goal point
    goal_pt = np.array([5.5, 5.5, 0.8])
    
    # Compute the reward
    reward, done, truncated = compute_exponential_reward(current_pose, goal_pt, client)
    
    # print(f"Reward: {reward}, Done: {done}, Truncated: {truncated}")
    
    if done or truncated:
        break
   
    
# Get the current car position
# current_pose = client.getCarState().kinematics_estimated.position
# print("Current car position: ", current_pose)
# Check if the target position is obstacle-free
# GetToNew(client)
# time.sleep(1)
# # Get the updated car position
# updated_pose = client.simGetVehiclePose()
# print("Updated car position: ", updated_pose.position)

# car_controls.throttle = 5
# car_controls.steering = 0
# client.setCarControls(car_controls)
# print("Go Forward")
# time.sleep(3)
# Disable API control
# client.enableApiControl(False)