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
        self.rl_delta_limit = 0.3  # rad

        # Normalized action space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation:
        # [lane_error, psi, v, dx_obs, dy_obs]
        obs_low = np.array([
            -10.0,                  # lane error
            self.model.psi_min,     # psi
            self.model.v_min,       # v
            -1e3,                   # dx_obs
            -1e3                    # dy_obs
        ], dtype=np.float32)

        obs_high = np.array([
            10.0,                   # lane error
            self.model.psi_max,     # psi
            self.model.v_max,       # v
            1e3,                    # dx_obs
            1e3                     # dy_obs
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
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

        y_ref = 2.0
        lane_error = y_pos - y_ref

        obs = np.array([
            lane_error,
            psi,
            v,
            dx_obs,
            dy_obs
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
        x, y, psi, v = next_state
        a, delta_f = action

        y_ref = 2.0
        reward = 0.0

        # 1. Forward progress
        reward += 2.0 * (next_state[0] - state[0])

        # 2. Stay near right-lane center
        reward -= 1.5 * (y - y_ref) ** 2

        # 3. Heading regularization
        reward -= 0.2 * psi ** 2

        # 4. Control effort penalty
        reward -= 0.01 * a ** 2
        reward -= 0.01 * delta_f ** 2

        # 5. Smoothness penalty
        reward -= 0.05 * (action[0] - prev_u[0]) ** 2
        reward -= 0.05 * (action[1] - prev_u[1]) ** 2

        # 6. Soft obstacle proximity penalty
        dist_obs = np.sqrt((x - self.model.obs_x) ** 2 + (y - self.model.obs_y) ** 2)
        if dist_obs < 8.0:
            reward -= 2.0 * (8.0 - dist_obs)

        # 7. Terminal penalties / bonus
        if collision:
            reward -= 300.0

        if out_of_highway:
            reward -= 300.0

        if state_violation:
            reward -= 200.0

        if reached_goal:
            reward += 100.0

        return float(reward)