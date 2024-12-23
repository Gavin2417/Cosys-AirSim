from gymnasium.envs.registration import register

register(
    id="airsim-drone-sample-v0", entry_point="airgym.envs:AirSimDroneEnv",
)

register(
    id="airsim-car-sample-v0", entry_point="airgym.envs:AirSimCarEnv",
)

register(
    id="airsim-car-lidar-sample-v0", entry_point="airgym.envs:AirSimCarEnvLidar",
)
