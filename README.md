# RL Obstacle Avoidance

A reinforcement learning agent trained to navigate a highway and avoid a static obstacle using a kinematic bicycle vehicle model.

The agent is trained with **Soft Actor-Critic (SAC)** from Stable-Baselines3 inside a custom Gymnasium environment. The vehicle dynamics are integrated with a CasADi-based RK4 integrator. The reward function is designed to mirror the cost structure of an NMPC controller solved on the same problem.

---

## Problem Setup

- **Highway**: 2-lane road, each lane 4 m wide (total width 8 m)
- **Vehicle**: kinematic bicycle model (state: `[x, y, psi, v]`, inputs: `[a, delta_f]`)
- **Obstacle**: static circle at `(100 m, 2 m)` with radius 1 m + 1.2 m safety margin
- **Start**: `x = 0 m`, `y = 2.5 m`, `v = 120 km/h`
- **Goal**: drive from `x = 0` to `x >= 170 m` without collision or leaving the road

---

## Project Structure

```
RL_ObstacleAvoidance/
├── vehicle_model.py         # Kinematic bicycle model + CasADi RK4 integrator
├── rl_env.py                # Custom Gymnasium environment (HighwayObstacleEnv + ActionSmoothingWrapper)
├── train_rl.py              # SAC training script with checkpointing
├── evaluate_rl.py           # Load best model and run an evaluation episode
├── rollout_rl_env.py        # Random-action rollout for environment sanity check
├── plot_utils.py            # Trajectory and state/input plotting helpers
├── test_rl_env.py           # Unit tests for the RL environment
├── test_vechile_model.py    # Unit tests for the vehicle model
├── models/
│   ├── best_model.zip                        # Best model saved during training
│   ├── sac_highway_obstacle.zip              # Final model after full training run
│   └── checkpoints/                          # Intermediate checkpoints (every 25k steps)
└── logs/
    ├── eval/evaluations.npz                  # Evaluation metrics from training
    └── sac_highway_tensorboard/              # TensorBoard training logs
```

---

## Observation Space

The agent receives a 7-dimensional normalized observation at each step:

| Index | Signal | Normalization |
| ----- | ------ | ------------- |
| 0 | Lane error `(y - 2.0)` | / lane width (4 m) |
| 1 | Heading `psi` | / (pi/2) |
| 2 | Speed `v` | / v_max |
| 3 | Longitudinal distance to obstacle `dx` | / obs_x (100 m) |
| 4 | Lateral distance to obstacle `dy` | / lane width (4 m) |
| 5 | Forward progress `x` | / obs_x (100 m) |
| 6 | Scalar distance to obstacle | / obs_x (100 m) |

## Action Space

Continuous, normalized to `[-1, 1]^2`. An `ActionSmoothingWrapper` applies EMA smoothing (alpha=0.7) before passing actions to the environment:

| Index | Physical quantity | Range |
| ----- | ---------------- | ----- |
| 0 | Acceleration `a` | -10 to +3 m/s^2 |
| 1 | Steering angle `delta_f` | -0.35 to +0.35 rad |

---

## Reward Function

The reward mirrors the NMPC cost function (MPC weights: `w_y=20`, `w_delta=20`, `w_obs=1000`, `w_acc_rate=100`, `w_steer_rate=50`):

| Component | Formula |
| --------- | ------- |
| Forward progress | `+1.0 * dx / (v_max * dt)` |
| Velocity tracking | `-1.0 * (v - v_max)^2` |
| Lane centering | `-lat_weight * (y - 2)^2` |
| Heading alignment | `-0.5 * psi^2` |
| Steering effort | `-0.1 * delta_f^2` |
| Acceleration rate | `-0.3 * (da / a_range)^2` |
| Steering rate | `-1.0 * (dd / d_limit)^2` |
| Obstacle penalty | `-100 / dist^2` |
| Lateral velocity | `-0.5 * (y_dot / v_max)^2` |
| Settling bonus | `+0.5` if `|y-2|<0.2`, `|psi|<0.05`, `|delta_f|<0.05` |
| Collision | `-100` |
| Out of road | `-100` |
| State violation | `-50` |
| Reached goal `x >= 170 m` | `+200` |

**Lane weight zones**: normal `lat_weight=1.0`; near obstacle (dist < obs_r_eff + 5 m) `lat_weight=0.05`.

---

## Installation

### Prerequisites

- Python 3.9+
- A working C/C++ compiler (required by CasADi)

### Install dependencies

```bash
pip install numpy matplotlib gymnasium stable-baselines3 casadi tensorboard
```

---

## Dependencies

| Package | Purpose |
| ------- | ------- |
| `numpy` | Numerical arrays and math |
| `matplotlib` | Trajectory and state/input plots |
| `gymnasium` | RL environment base class |
| `stable-baselines3` | SAC algorithm, Monitor, EvalCallback, CheckpointCallback |
| `casadi` | Symbolic vehicle dynamics and RK4 integrator |
| `tensorboard` | Training curve visualization |

---

## Usage

### Train the agent

```bash
python train_rl.py
```

Training runs for 150,000 timesteps. Checkpoints are saved every 25,000 steps to `models/checkpoints/`. The best model (by mean evaluation reward) is saved to `models/best_model.zip`.

### Evaluate the trained agent

```bash
python evaluate_rl.py
```

Loads `models/best_model.zip` (or `models/sac_highway_obstacle.zip` as fallback), runs one episode, prints the total reward, and saves plots to `logs/trajectory.png` and `logs/states_inputs.png`.

### Random rollout (sanity check)

```bash
python rollout_rl_env.py
```

Runs the environment with random actions and displays trajectory and state/input plots.

### Monitor training with TensorBoard

```bash
tensorboard --logdir logs/sac_highway_tensorboard
```

---

## Vehicle Model Parameters

| Parameter | Value |
| --------- | ----- |
| Car length `l_car` | 3.0 m |
| Car width `w_car` | 2.0 m |
| Wheelbase `lf = lr` | 1.5 m |
| Time step `dt` | 0.05 s |
| Speed range | 0 - 36.1 m/s (0 - 130 km/h) |
| Acceleration range | -10 to +3 m/s^2 |
| Steering limit (RL) | -0.35 to +0.35 rad |
| Initial speed | 120 km/h (33.3 m/s) |
| Initial position | x=0 m, y=2.5 m |