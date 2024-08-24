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
# client.enableApiControl(True)
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
        new_position = airsim.Vector3r(0, 0, current_pose.position.z_val)
        print("New position: ", new_position)

        orientation = airsim.Quaternionr(0, 0, 0)
        new_pose = airsim.Pose(new_position, orientation)
        client.simSetVehiclePose(new_pose, ignore_collision=False)
        
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
GetToNew(client)

# Get the current car position
current_pose = client.simGetVehiclePose()
print("Current car position: ", current_pose.position)

# Goal point 
goal = airsim.Vector3r(4.5, 4.5 ,current_pose.position.z_val)
goal_pose = airsim.Pose(goal, airsim.Quaternionr(0,0,0))

time.sleep(1)
# Set the car controls
# car_controls.steering = 0
# car_controls.throttle = 1
# client.setCarControls(car_controls)
while True:
    
    current_pose = client.simGetVehiclePose()
    print("Goal position: ", goal.to_numpy_array())
    print("Current position: ", current_pose.position.to_numpy_array())
    print("Distance to goal1: ", euclidean_distance(goal, current_pose.position))
    # if euclidean_distance(goal, current_pose.position) <= 4.6:
    #     break
    time.sleep(0.5)

#restore to original state
# client.reset()

client.enableApiControl(False)