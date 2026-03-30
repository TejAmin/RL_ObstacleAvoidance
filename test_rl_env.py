import numpy as np
from rl_env import HighwayObstacleEnv


env = HighwayObstacleEnv(max_steps=120)

obs, info = env.reset()
print("Initial observation:", obs)

for k in range(20):
    # random normalized action
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nStep {k+1}")
    print("Action:", action)
    print("Obs:", obs)
    print("Reward:", reward)
    print("Info:", info)

    if terminated or truncated:
        print("Episode ended.")
        break