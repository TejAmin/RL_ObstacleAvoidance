import numpy as np
import gymnasium as gym
from gymnasium import spaces

from vehicle_model import VehicleModel


class HighwayObstacleEnv(gym.Env):
    """
    RL environment for highway obstacle avoidance based on the same
    kinematic bicycle model used in the MPC assignment.

    Internal state:
        x = [x_pos, y_pos, psi, v]

    Agent action (normalized):
        action = [a_norm, delta_norm] in [-1, 1]^2

    Physical input:
        u = [a, delta_f]

    Observation returned to agent:
        obs = [lane_error, psi, v, dx_obs, dy_obs]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=120):
        super().__init__()

        self.model = VehicleModel()
        self.integrator = self.model.create_collocation_integrator()

        self.max_steps = max_steps
        self.step_count = 0

        self.state = None
        self.prev_u = None

        # Practical steering limit for RL training
        self.rl_delta_limit = 0.2 # rad

        # Normalized action space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation (normalized): 7 elements
        # [lane_error_norm, psi_norm, v_norm, dx_obs_norm, dy_obs_norm, x_progress_norm, dist_obs_norm]
        self.observation_space = spaces.Box(
            low=np.full(7, -5.0, dtype=np.float32),
            high=np.full(7, 5.0, dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = self.model.x0.copy()

     # small initial perturbations
        self.state[1] += np.random.uniform(-0.2, 0.2)   # y
        self.state[2] += np.random.uniform(-0.02, 0.02) # psi

        self.prev_u = self.model.u0.copy()
        self.step_count = 0

        obs = self._get_obs(self.state)
        info = {}

        return obs, info

    def step(self, action):
        self.step_count += 1

        # Convert normalized RL action to physical input
        u = self._scale_action(action)

        # Simulate one step using the same collocation-based model
        res = self.integrator(x0=self.state, p=u)
        next_state = np.array(res["xf"]).reshape(-1)

        # Clip bounded states
        next_state[2] = np.clip(next_state[2], self.model.psi_min, self.model.psi_max)
        next_state[3] = np.clip(next_state[3], self.model.v_min, self.model.v_max)

        # Checks
        collision = self._check_collision(next_state)
        out_of_highway = self._check_out_of_highway(next_state)
        state_violation = self._check_state_violation(next_state)
        reached_goal = next_state[0] >= 110.0
        timeout = self.step_count >= self.max_steps

        terminated = collision or out_of_highway or state_violation or reached_goal
        truncated = timeout and not terminated

        reward = self._compute_reward(
            state=self.state,
            action=u,
            next_state=next_state,
            prev_u=self.prev_u,
            collision=collision,
            out_of_highway=out_of_highway,
            state_violation=state_violation,
            reached_goal=reached_goal
        )

        self.state = next_state
        self.prev_u = u

        obs = self._get_obs(self.state)

        info = {
            "collision": collision,
            "out_of_highway": out_of_highway,
            "state_violation": state_violation,
            "reached_goal": reached_goal,
            "u_physical": u.copy()
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self, x):
        x_pos, y_pos, psi, v = x

        dx_obs = self.model.obs_x - x_pos
        dy_obs = self.model.obs_y - y_pos
        dist_obs = np.sqrt(dx_obs**2 + dy_obs**2)

        y_ref = 2.0
        lane_width = self.model.lane_width  # 4.0 m

        obs = np.array([
            (y_pos - y_ref) / lane_width,           # lane error, normalized
            psi / (np.pi / 2),                      # heading, normalized
            v / self.model.v_max,                   # speed, normalized
            dx_obs / self.model.obs_x,              # longitudinal dist to obs, normalized
            dy_obs / lane_width,                    # lateral dist to obs, normalized
            x_pos / self.model.obs_x,              # forward progress [0, 1+]
            dist_obs / self.model.obs_x,            # scalar dist to obstacle, normalized
        ], dtype=np.float32)

        return obs

    def _scale_action(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        action = np.clip(action, -1.0, 1.0)

        # Make normalized 0 correspond to physical 0
        if action[0] >= 0.0:
            a = action[0] * self.model.a_max
        else:
            a = action[0] * abs(self.model.a_min)

        delta_f = action[1] * self.rl_delta_limit

        u = np.array([a, delta_f], dtype=float)
        return self.model.clip_input(u)

    def _check_out_of_highway(self, x):
        y = x[1]
        return (y < self.model.road_y_min) or (y > self.model.road_y_max)

    def _check_state_violation(self, x):
        psi = x[2]
        v = x[3]

        return (
            (psi < self.model.psi_min) or
            (psi > self.model.psi_max) or
            (v < self.model.v_min) or
            (v > self.model.v_max)
        )

    def _check_collision(self, x):
        px, py = x[0], x[1]

        dist = np.sqrt((px - self.model.obs_x) ** 2 + (py - self.model.obs_y) ** 2)

        # First RL version: point-mass vehicle with obstacle safety margin
        safe_radius = self.model.obs_r + self.model.obs_margin

        return dist <= safe_radius
    def _compute_reward(
        self,
        state,
        action,
        next_state,
        prev_u,
        collision,
        out_of_highway,
        state_violation,
        reached_goal
    ):
        x_prev, _, _, _ = state
        x, y, psi, _ = next_state
        a, delta_f = action

        y_ref = 2.0
        lane_width = self.model.lane_width  # 4.0 m

        dx = x - self.model.obs_x
        dy = y - self.model.obs_y
        dist = np.sqrt(dx**2 + dy**2)

        reward = 0.0

        # 1. Forward progress reward (normalized, ~[0, 1] per step)
        dx_progress = x - x_prev
        reward += dx_progress / (self.model.v_max * self.model.dt)

        # 2. Lane-centering penalty (normalized by lane width, near-zero when centered)
        lane_error_norm = (y - y_ref) / lane_width
        # Reduce lane penalty near obstacle so agent can swerve
        near_obs = dist < 15.0
        lane_weight = 0.05 if near_obs else 0.5
        reward -= lane_weight * lane_error_norm**2

        # 3. Heading penalty (discourage large yaw)
        reward -= 0.1 * (psi / (np.pi / 2))**2

        # 4. Action smoothness — penalize magnitude and rate of change for both inputs
        a_range = self.model.a_max - self.model.a_min   # 13.0 m/s²
        reward -= 0.1 * (a / a_range)**2
        reward -= 0.3 * ((a - prev_u[0]) / a_range)**2
        reward -= 0.1 * (delta_f / self.rl_delta_limit)**2
        reward -= 0.3 * ((delta_f - prev_u[1]) / self.rl_delta_limit)**2

        # 5. Obstacle proximity: exponential penalty, active within 30m
        if dist < 30.0:
            sigma = 8.0
            reward -= 3.0 * np.exp(-dist / sigma)

        # 6. Terminal penalties / bonus (dominant signals)
        if collision:
            reward -= 100.0

        if out_of_highway:
            reward -= 100.0

        if state_violation:
            reward -= 50.0

        if reached_goal:
            reward += 200.0

        return float(reward)