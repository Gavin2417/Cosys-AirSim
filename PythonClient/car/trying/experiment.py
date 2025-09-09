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

# Keep current Z height as baseline
saved_z = client.simGetVehiclePose().position.z_val

# Scenarios: strictly start -> goal
# dz applies ONLY to the start Z as an offset from saved_z
scenarios = [
    # # {"label": "goal", "start": (12, -8), "goal": (-5, -1)},                       # goal : (17,-7) -> (-5,-1)
    {"label": "normal", "start": (-30, 7), "goal": (-21.5, 17)},                  # Normal
    {"label": "uneven", "start": (-30, -20), "goal": (-21.5, -6), "dz": -0.5},    # uneven (dz = -0.5)
    {"label": "ramp", "start": (-31, -44), "goal": (-20, -37)},                   # ramp
    {"label": "two_height_ramp", "start": (-55, 5), "goal": (-41, 5)},            # two height ramp
    {"label": "ramp_obstacle", "start": (-33, 40), "goal": (-20, 40)},            # ramp obstacle
    {"label": "hole", "start": (-55.2, -13), "goal": (-41, -13), "dz": -4.5},     # hole (dz = -4.5)
]

# Create a log file with timestamp
log_file = os.path.join(os.path.dirname(__file__), f"step_randla_log.txt")

# Helper: run a script with given goal, and log output
def run_script(script_path, goal_x, goal_y, log_handle, label=""):
    result = subprocess.run(
        ["python3", script_path, "--xgoal", str(goal_x), "--ygoal", str(goal_y), "--name", label],
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

    for sc in scenarios:
        start = sc["start"]
        goal = sc["goal"]
        label = sc.get("label", "scenario")
        dz = sc.get("dz", 0.0)

        print(f"\n=== Starting scenario '{label}': {start} -> {goal} (dz={dz}) ===")
        for run in range(1, NUM_RUNS + 1):
            run_header = f"\n=== Run {run} | [{label}] Start: {start} -> Goal: {goal} | dz={dz} ==="
            print(run_header)
            log.write(run_header + "\n")

            # 1. Reset position to start (apply dz to saved_z only for start)
            start_z = saved_z + dz
            start_pose = airsim.Pose(
                airsim.Vector3r(start[0], start[1], start_z),
                airsim.Quaternionr(0, 0, 0, 1)
            )
            client.setCarControls(airsim.CarControls(throttle=0, steering=0), 'CPHusky')
            client.simSetVehiclePose(start_pose, ignore_collision=True)
            time.sleep(0.1)

            # 2. Run test_step
            print("Running test_step.py...")
            run_script(SCRIPT_STEP, goal[0], goal[1], log, label=f"{label}_{run}")

            # Reset to the same start before test_randla
            client.setCarControls(airsim.CarControls(throttle=0, steering=0), 'CPHusky')
            client.simSetVehiclePose(start_pose, ignore_collision=True)
            time.sleep(0.1)

            # 3. Run test_randla
            print("Running test_randla.py...")
            run_script(SCRIPT_RANDLA, goal[0], goal[1], log, label=f"{label}_{run}")

    log.write(f"\nAll runs complete at {datetime.now()}\n")
    print("All runs complete.")

client.enableApiControl(False)