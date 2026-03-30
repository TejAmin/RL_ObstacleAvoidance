import numpy as np
from rl_env import HighwayObstacleEnv
from plot_utils import plot_trajectory, plot_states_and_inputs

env = HighwayObstacleEnv(max_steps=120)

obs, _ = env.reset()

states = [env.state.copy()]
inputs = []

for k in range(120):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    states.append(env.state.copy())
    inputs.append(info["u_physical"].copy())

    if terminated or truncated:
        print(f"Episode ended at step {k+1}")
        print(info)
        break

states = np.array(states)
inputs = np.array(inputs)

plot_trajectory(states, env.model, show=True)
plot_states_and_inputs(states, inputs, env.model.dt, show=True)