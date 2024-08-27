import setup_path
import gymnasium
import airgym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

def custom_evaluate(model, env, n_eval_episodes=10):
    success_count = 0
    fail_count = 0
    collision_count = 0
    episode_rewards = []
    episode_lengths = []
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        length = 0
        while not done:
            # Predict action using the trained model
            action, _states = model.predict(obs, deterministic=True)
            # Take action in the environment
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Check for collision in the info dictionary

            # Check for success or failure (done + success/failure criteria)
            if done:
                if reward == 20:
                    success_count += 1
                elif reward == -10:
                    collision_count += 1
                    fail_count += 1
                else:
                    fail_count += 1
            length += 1
        print(f"Episode {episode + 1}: {total_reward}, Success: {success_count}, Fail: {fail_count}, Collision: {collision_count},reward: {reward}")
        episode_rewards.append(total_reward)
        episode_lengths.append(length) 

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_lenght = np.mean(episode_lengths)
    std_lenght = np.std(episode_lengths)

    # Print out the evaluation results
    print("Evaluation Results")
    print("=================================")
    print(f"Evaluation Results over {n_eval_episodes} episodes:")
    print(f"Mean time steps: {mean_lenght:.2f} +/- {std_lenght:.2f}")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Successes: {success_count}")
    print(f"Failures: {fail_count}")
    print(f"Collisions: {collision_count}")

    return mean_reward, std_reward, success_count, fail_count, collision_count

# Create a DummyVecEnv for the airsim gym environment
env = DummyVecEnv(
    [
        lambda: gymnasium.make(
            "airgym:airsim-car-lidar-sample-v0",
            ip_address="127.0.0.1",
            image_shape=(84, 84, 1),
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Load the trained PPO model
model = PPO.load("ppo_airsim_car_policy")

# Evaluate the model using the custom evaluation function
custom_evaluate(model, env, n_eval_episodes=1000)
