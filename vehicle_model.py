import casadi as ca
import numpy as np


class VehicleModel:
    """
    Kinematic bicycle model for the highway obstacle avoidance project.

    State:
        x = [x_pos, y_pos, psi, v]

    Input:
        u = [a, delta_f]
    """

    def __init__(self):
        # Vehicle / environment parameters from the project sheet
        self.l_car = 3.0
        self.w_car = 2.0
        self.lf = self.l_car / 2.0
        self.lr = self.l_car / 2.0

        self.lane_width = 4.0
        self.num_lanes = 2
        self.road_y_min = 0.0
        self.road_y_max = self.num_lanes * self.lane_width

        self.obs_x = 100.0
        self.obs_y = 2.0
        self.obs_r = 1.0
        self.obs_margin = 1.2

        self.dt = 0.05

        # Bounds from the sheet
        self.psi_min = -np.pi / 2.0
        self.psi_max = np.pi / 2.0
        self.v_min = 0.0
        self.v_max = 130.0 / 3.6

        self.a_min = -10.0
        self.a_max = 3.0
        self.delta_min = -np.pi / 2.0
        self.delta_max = np.pi / 2.0

        # Initial condition from the sheet
        self.x0 = np.array([0.0, 2.5, 0.0, 120.0 / 3.6], dtype=float)
        self.u0 = np.array([0.0, 0.0], dtype=float)

    @staticmethod
    def _to_numpy(vec) -> np.ndarray:
        return np.array(vec, dtype=float).reshape(-1)

    def clip_input(self, u) -> np.ndarray:
        u = self._to_numpy(u)
        u[0] = np.clip(u[0], self.a_min, self.a_max)
        u[1] = np.clip(u[1], self.delta_min, self.delta_max)
        return u

    def beta_symbolic(self, delta_f):
        return ca.atan((self.lr / (self.lf + self.lr)) * ca.tan(delta_f))

    def beta_numeric(self, delta_f: float) -> float:
        return float(np.arctan((self.lr / (self.lf + self.lr)) * np.tan(delta_f)))

    def continuous_dynamics_symbolic(self, x, u):
        """
        Symbolic continuous-time dynamics for CasADi integrator.
        """
        x_pos, y_pos, psi, v = x[0], x[1], x[2], x[3]
        a, delta_f = u[0], u[1]

        beta = self.beta_symbolic(delta_f)

        x_dot = v * ca.cos(psi + beta)
        y_dot = v * ca.sin(psi + beta)
        psi_dot = (v / self.lr) * ca.sin(beta)
        v_dot = a

        return ca.vertcat(x_dot, y_dot, psi_dot, v_dot)

    def continuous_dynamics_numeric(self, x, u) -> np.ndarray:
        """
        Numeric version, useful for quick debugging if needed.
        """
        x = self._to_numpy(x)
        u = self.clip_input(u)

        x_pos, y_pos, psi, v = x
        a, delta_f = u

        beta = self.beta_numeric(delta_f)

        x_dot = v * np.cos(psi + beta)
        y_dot = v * np.sin(psi + beta)
        psi_dot = (v / self.lr) * np.sin(beta)
        v_dot = a

        return np.array([x_dot, y_dot, psi_dot, v_dot], dtype=float)

    def create_casadi_ode(self) -> ca.Function:
        """
        Returns a CasADi function f(x, u) = x_dot
        """
        x = ca.MX.sym("x", 4)
        u = ca.MX.sym("u", 2)
        xdot = self.continuous_dynamics_symbolic(x, u)
        return ca.Function("f", [x, u], [xdot])

    def create_collocation_integrator(self) -> ca.Function:
        """
        Orthogonal collocation on finite elements over one sample time dt.
        """
        x = ca.MX.sym("x", 4)
        u = ca.MX.sym("u", 2)

        dae = {
            "x": x,
            "p": u,
            "ode": self.continuous_dynamics_symbolic(x, u),
        }

        # CasADi collocation integrator over one control interval
        F = ca.integrator(
            "F",
            "collocation",
            dae,
            {
                "tf": self.dt,
                "number_of_finite_elements": 1,
                "interpolation_order": 3,
                "rootfinder": "newton",
            },
        )
        return F