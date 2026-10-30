#!/usr/bin/env python3
import sys
import time

import cosysairsim as airsim  # keep using the same fork you built the sim with

IP = "192.168.68.101"   # ← change if your sim runs on another box
PORT = 41451       # default AirSim RPC port

def main():
    print(f"Connecting to AirSim at {IP}:{PORT} ...")
    client = airsim.CarClient(ip=IP)  # or airsim.VehicleClient(ip=IP, port=PORT)
    # Retry loop in case the sim just launched
    for i in range(10):
        try:
            client.confirmConnection()
            print("✅ Connected")
            print("Ping:", client.ping())  # quick RPC sanity test
            pose = client.simGetVehiclePose()
            print("Vehicle pose:", pose)
            return 0
        except Exception as e:
            print(f"Attempt {i+1}/10 failed: {e}")
            time.sleep(1.0)
    print("❌ Could not connect. Check IP/port, firewall, settings.json, and sim is running.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
