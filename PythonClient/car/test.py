import setup_path
import cosysairsim as airsim
import numpy as np
import os
import time
import tempfile
import math
# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()


# Calculate Euclidean distance
def euclidean_distance(vec1, vec2):
    return math.sqrt(
        (vec2.x_val - vec1.x_val) ** 2 +
        (vec2.y_val - vec1.y_val) ** 2 +
        (vec2.z_val - vec1.z_val) ** 2
    )

def GetToNew(client):
    # Save current pose
    current_pose = client.simGetVehiclePose()
    
    while True:
        # Move to test position
        new_position = airsim.Vector3r(2, 0, current_pose.position.z_val)
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

# Check if the target position is obstacle-free
# GetToNew(client)

# # Get the current car position
# current_pose = client.simGetVehiclePose()
# print("Current car position: ", current_pose.position)


# time.sleep(1)
# # Set the car controls
# car_controls.steering = 0
# car_controls.throttle = 1
# client.setCarControls(car_controls)
while True:
    
    collision_info = client.simGetCollisionInfo()
    if collision_info.has_collided:
        # log the collision info
        print("Collision at: ", collision_info)
        # GetToNew(client)
        

        break

#restore to original state
# client.reset()

client.enableApiControl(False)