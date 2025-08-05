#!/usr/bin/env python3
import subprocess
import time
import os
import cosysairsim as airsim
from datetime import datetime

# number of runs per start-goal pair
NUM_RUNS = 5

# paths to your test scripts
SCRIPT_STEP = os.path.join(os.path.dirname(__file__), "test_step.py")
SCRIPT_RANDLA = os.path.join(os.path.dirname(__file__), "test_randla.py")

# connect once to AirSim
client = airsim.CarClient(ip="100.123.124.47")
client.confirmConnection()
client.enableApiControl(True)

# Keep current Z height for consistency
saved_z = client.simGetVehiclePose().position.z_val

# Start/goal points
points = [
    (5, -10),
    (0, 0),
    (17, -7),
    (0, -7),
    (9, 5)
]
RESUME_FROM = {
    "start": (0, -7),
    "goal": (17, -7),
    "run": 5  # will start from Run 4 (i.e., skip 1-3)
}

resume_reached = False
# Create a log file with timestamp
log_file = os.path.join(os.path.dirname(__file__), f"step_randla_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Helper: run a script with given goal, and log output
def run_script(script_path, goal_x, goal_y, log_handle, label=""):
    result = subprocess.run(
        ["python3", script_path, "--xgoal", str(goal_x), "--ygoal", str(goal_y)],
        capture_output=True,
        text=True
    )
    header = f"\n===== [{label}] Goal: ({goal_x}, {goal_y}) | Script: {os.path.basename(script_path)} =====\n"
    log_handle.write(header)
    log_handle.write(result.stdout)
    if result.returncode != 0:
        error_msg = f"ERROR (Return code {result.returncode}):\n{result.stderr}\n"
        print(error_msg)
        log_handle.write(error_msg)
    log_handle.flush()
    print(result.stdout)

# Loop and log
with open(log_file, "w") as log:
    log.write("=== STEP & RANDLA Test Log ===\n")
    log.write(f"Started at: {datetime.now()}\n\n")

    for start in points:
        for goal in points:
            
            if start == goal:
                continue
            print(f"\n=== Starting tests from {start} to {goal} ===")
            for run in range(1, NUM_RUNS + 1):

                if not resume_reached:
                    if (start, goal, run) == (RESUME_FROM["start"], RESUME_FROM["goal"], RESUME_FROM["run"]):
                        resume_reached = True
                    else:
                        continue
                run_header = f"\n=== Run {run} | Start: {start} -> Goal: {goal} ==="
                print(run_header)
                log.write(run_header + "\n")

                # 1. Reset position to start
                # Re set vehicle spped 
                start_pose = airsim.Pose(
                    airsim.Vector3r(start[0], start[1], saved_z),
                    airsim.Quaternionr(0, 0, 0, 1)
                )
                client.setCarControls(airsim.CarControls(throttle=0, steering=0), 'CPHusky')
                client.simSetVehiclePose(start_pose, ignore_collision=True)
                time.sleep(0.1)

                # 2. Run test_step
                print("Running test_step.py...")
                run_script(SCRIPT_STEP, goal[0], goal[1], log, label="test_step")
                
                client.setCarControls(airsim.CarControls(throttle=0, steering=0), 'CPHusky')
                client.simSetVehiclePose(start_pose, ignore_collision=True)
                time.sleep(0.1)

                # 3. Run test_randla
                print("Running test_randla.py...")
                run_script(SCRIPT_RANDLA, goal[0], goal[1], log, label="test_randla")

    log.write(f"\nAll runs complete at {datetime.now()}\n")
    print("All runs complete.")

client.enableApiControl(False)
