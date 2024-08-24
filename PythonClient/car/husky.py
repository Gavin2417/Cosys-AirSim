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

# Connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()


# Get the current car position
# current_pose = client.simGetVehiclePose()
# print("Current car position: ", current_pose.position)

# Check if the target position is obstacle-free
GetToNew(client)
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
client.enableApiControl(False)