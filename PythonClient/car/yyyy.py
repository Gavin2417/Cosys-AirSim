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

# # spawn the cube in the air
# client.simSpawnObject("Cube", "cube", airsim.Vector3r(-1, -1, 0), airsim.Quaternionr(0.01, 0.01,0.01))

# gET ALL OBJECTS position
# all_objects = client.simListSceneObjects()
# for obj in all_objects:
    
#     obj_pose = client.simGetObjectPose(obj)
#     print(f"Object: {obj}, Pose: {obj_pose}")

# get the pose of the vehicle
vehicle_pose = client.simGetVehiclePose()
print(f"Vehicle pose: {vehicle_pose}")

# Get lidar objectr position
lidar_pose = client.simGetObjectPose("CPHusky_gpulidar_gpulidar1")
print(f"Lidar pose: {lidar_pose}")

# Get lidar objectr position
new_pose = client.simGetObjectPose("Cylinder123")
print(f"Lidar pose: {new_pose}")