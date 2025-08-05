#!/usr/bin/env python3
import subprocess
import time
import os
import numpy as np
import cosysairsim as airsim

# number of runs you want
NUM_RUNS = 10

# fixed spawn


# path to your step script
SCRIPT = os.path.join(os.path.dirname(__file__), "test_step.py")

# connect once, outside the loop
client = airsim.CarClient(ip="100.123.124.47")
current_pose = client.simGetVehiclePose()
saved_z = current_pose.position.z_val
SPAWN_POSE = airsim.Pose(
    airsim.Vector3r(0, 0, current_pose.position.z_val),
    airsim.Quaternionr(0, 0, 0, 1)
)
client.confirmConnection()
client.enableApiControl(True)

for run in range(1, NUM_RUNS+1):
    print(f"\n=== Run {run}/{NUM_RUNS} ===")

    # 1) teleport to spawn
    client.simSetVehiclePose(SPAWN_POSE, ignore_collision=True)
    time.sleep(0.1)   # give the sim a tick

    result = subprocess.run(
        ["python3", SCRIPT, "--xgoal", str(xgoal), "--ygoal", str(ygoal)],
        capture_output=True,
        text=True
    )

    # print any stdout/stderr from the child
    print(result.stdout)
    if result.returncode != 0:
        print("Child script failed with:", result.stderr)
        break

client.enableApiControl(False)
print("All runs complete.")
