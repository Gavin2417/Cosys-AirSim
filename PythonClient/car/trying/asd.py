#!/usr/bin/env python3
import subprocess
import time
import os
import argparse
import numpy as np
import cosysairsim as airsim

# ---- Argument parser ----
parser = argparse.ArgumentParser(description="AirSim multi-run test")
parser.add_argument("--start_x", type=float, default=-5.0,
                    help="Spawn X position (meters)")
parser.add_argument("--start_y", type=float, default=-20.0,
                    help="Spawn Y position (meters)")
args = parser.parse_args()

# number of runs you want
NUM_RUNS = 10

# path to your step script
SCRIPT = os.path.join(os.path.dirname(__file__), "test_step.py")

# connect once, outside the loop
client = airsim.CarClient(ip="100.123.124.47")
current_pose = client.simGetVehiclePose()
saved_z = current_pose.position.z_val

# use argparse values for spawn position
SPAWN_POSE = airsim.Pose(
    airsim.Vector3r(args.start_x, args.start_y, saved_z-4.5),
    airsim.Quaternionr(0, 0, 0, 1)
)

client.confirmConnection()
client.enableApiControl(True)
client.simSetVehiclePose(SPAWN_POSE, ignore_collision=True)

# your loop or subprocess call would go here
# for run in range(1, NUM_RUNS+1):
#     ...

client.enableApiControl(False)
print(f"Spawned at X={args.start_x}, Y={args.start_y}. All runs complete.")
