import setup_path
import gymnasium
import airgym
import time
import wandb

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

# Initialize wandb
# wandb.init(project="dqn_airsim_car", sync_tensorboard=True)

class WandbCallback(BaseCallback):
    def _on_step(self) -> bool:
        logs = {"timesteps": self.num_timesteps}

        # Log rewards if available
        if "rewards" in self.locals:
            logs["reward"] = self.locals["rewards"].item()
        
        # Log done status if available
        if "dones" in self.locals:
            logs["done"] = int(self.locals["dones"].item())
        
        # Log additional information if available
        if "infos" in self.locals and len(self.locals["infos"]) > 0:
            info = self.locals["infos"][0]
            if "collision" in info:
                logs["collision"] = int(info["collision"])
            if "TimeLimit.truncated" in info:
                logs["truncated"] = int(info["TimeLimit.truncated"])
            
            # Log episode information if available
            if "episode" in info:
                logs["episode_reward"] = info["episode"]["r"]
                logs["episode_length"] = info["episode"]["l"]

        wandb.log(logs)
        return True

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gymnasium.make(
                "airgym:airsim-car-lidar-sample-v0",
                ip_address="127.0.0.1",
                image_shape=(84, 84, 1),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    batch_size=32,
    device="cuda",
    tensorboard_log="./tb_logs/",
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=5000,
)
callbacks.append(eval_callback)

# # Add wandb callback
# wandb_callback = WandbCallback()
# callbacks.append(wandb_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=6e5, 
    tb_log_name="ppo_airsim_car_run_" + str(time.time()), 
    log_interval=1,  # Logs every 10 iterations
    **kwargs
)

# Save policy weights
model.save("ppo_airsim_car_policy")

# Finish wandb run
# wandb.finish()