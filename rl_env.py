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
        self.rl_delta_limit = 0.35  # rad

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
        reached_goal = next_state[0] >= 170.0
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
        x, y, psi, v = next_state
        a, delta_f = action
        prev_a, prev_delta = prev_u

        y_ref   = 2.0
        a_range = self.model.a_max - self.model.a_min  # 13.0 m/s²

        # --- Obstacle geometry ---
        obs_r_eff = self.model.obs_r + self.model.obs_margin          # 2.2 m
        dx_obs  = x - self.model.obs_x
        dy_obs  = y - self.model.obs_y
        dist_sq = dx_obs ** 2 + dy_obs ** 2

        # --- MPC Term 1: Velocity maximization  (MPC: minimize -0.5*v) ---
        v_error = (v - self.model.v_max) ** 2   # 7.84 at v=33.3 → strong bang-coast

        # --- MPC Term 2: Lateral tracking y=2  (MPC: w_y*(y-2)², weight_factor near obs) ---
        lat_error  = (y - y_ref) ** 2
        # Reduce lateral pull when close to obstacle so agent swerves freely (MPC: 0.1×)
        lat_weight = 0.05 if dist_sq < (obs_r_eff + 5.0) ** 2 else 1.0

        # --- MPC Term 3: Heading alignment ---
        heading_error = psi ** 2

        # --- MPC Term 4: Steering effort  (MPC: w_delta*delta_f²) ---
        steer_effort = delta_f ** 2

        # --- MPC Term 5: Obstacle 1/dist²  (MPC: w_obs/dist_sq = 1000/dist²) ---
        dist_sq_safe   = max(float(dist_sq), obs_r_eff ** 2)
        obstacle_cost  = 100.0 / dist_sq_safe  # ~5.7 at 10 m, ~22 at 5 m, ~100 at 2.2 m (edge)

        # --- MPC Term 6: Acceleration rate  (MPC: w_acc_rate*(Δa)²) ---
        accel_rate  = ((a - prev_a) / a_range) ** 2

        # --- MPC Term 7: Steering rate  (MPC: w_steer_rate*(Δdelta)²) ---
        steer_rate  = ((delta_f - prev_delta) / self.rl_delta_limit) ** 2

        # --- RL-specific: Forward progress (MPC has finite horizon; RL needs explicit signal) ---
        progress = (x - x_prev) / (self.model.v_max * self.model.dt)

        # --- RL-specific: Lateral velocity — damps overshoot/undershoot ---
        y_dot       = (y - state[1]) / self.model.dt
        lat_vel     = (y_dot / self.model.v_max) ** 2

        # Settling bonus: explicit positive reward for stable cruising at lane center
        settling = 5.0 if (
            abs(y - y_ref) < 0.2 and
            abs(psi) < 0.05 and
            abs(delta_f) < 0.05
        ) else 0.0

        reward = (
            + 1.0 * progress
            - 1.0 * v_error            # MPC: -0.5*v  → bang-coast to v_max
            - lat_weight * lat_error   # MPC: 20*(y-2)², reduced near obstacle
            - 0.5 * heading_error      # MPC: implicit via dynamics
            - 0.1 * steer_effort       # MPC: w_delta=20 * delta_f²
            - 0.3 * accel_rate         # MPC: w_acc_rate=100 * (Δa)²
            - 1.0 * steer_rate         # MPC: w_steer_rate=50 * (Δdelta)²
            - obstacle_cost            # MPC: w_obs=1000 / dist²
            - 0.5 * lat_vel            # RL: dampen lateral oscillations
            + settling                 # bonus for stable cruising at lane center
        )

        # Terminal signals (dominant — RL-specific)
        if collision:
            reward -= 100.0
        if out_of_highway:
            reward -= 100.0
        if state_violation:
            reward -= 50.0
        if reached_goal:
            reward += 200.0

        return float(reward)


class ActionSmoothingWrapper(gym.Wrapper):
    """EMA-smooths normalized actions before passing to the base env."""

    def __init__(self, env, alpha: float = 0.7):
        super().__init__(env)
        self.alpha = alpha
        self._prev_action = np.zeros(env.action_space.shape, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Initialize at max accel so first smoothed action hits ~3.0 m/s² immediately
        self._prev_action = np.array([1.0, 0.0], dtype=np.float32)
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        smoothed = self.alpha * action + (1.0 - self.alpha) * self._prev_action
        smoothed = np.clip(smoothed, self.action_space.low, self.action_space.high)
        self._prev_action = smoothed.copy()
        return self.env.step(smoothed)