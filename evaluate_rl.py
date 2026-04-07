import numpy as np
from stable_baselines3 import SAC

from rl_env import HighwayObstacleEnv
from plot_utils import plot_trajectory, plot_states_and_inputs


def main():
    env = HighwayObstacleEnv(max_steps=120)
    import os
    model_path = "models/best_model" if os.path.exists("models/best_model.zip") else "models/sac_highway_obstacle"
    print(f"Loading model from: {model_path}")
    model = SAC.load(model_path)

    obs, _ = env.reset()

    states = [env.state.copy()]
    inputs = []
    rewards = []

    for k in range(120):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        states.append(env.state.copy())
        inputs.append(info["u_physical"].copy())
        rewards.append(reward)

        if terminated or truncated:
            print(f"Episode ended at step {k+1}")
            print(info)
            break

    states = np.array(states)
    inputs = np.array(inputs)

    os.makedirs("logs", exist_ok=True)
    plot_trajectory(states, env.model, show=False, save_path="logs/trajectory.png")
    plot_states_and_inputs(states, inputs, env.model.dt, show=False, save_path="logs/states_inputs.png")

    print("Total reward:", np.sum(rewards))
    print("Plots saved to logs/trajectory.png and logs/states_inputs.png")
    env.close()


if __name__ == "__main__":
    main()