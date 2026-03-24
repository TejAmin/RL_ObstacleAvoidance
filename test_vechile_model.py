import numpy as np

from vehicle_model import VehicleModel
from plot_utils import plot_trajectory, plot_states_and_inputs


def simulate_model_test():
    """
    Mandatory model test from the assignment:
    - discretize with orthogonal collocation on finite elements
    - dt = 0.05 s
    - Nsim = 100
    - first half:  u1 = [0.1,  5e-4*pi]
    - second half: u2 = [-0.1, -5e-4*pi]
    """
    model = VehicleModel()
    F = model.create_collocation_integrator()

    Nsim = 100
    xk = model.x0.copy()

    u1 = np.array([0.1, 5e-4 * np.pi], dtype=float)
    u2 = np.array([-0.1, -5e-4 * np.pi], dtype=float)

    states = [xk.copy()]
    inputs = []

    for k in range(Nsim):
        uk = u1 if k < Nsim // 2 else u2
        uk = model.clip_input(uk)

        inputs.append(uk.copy())

        res = F(x0=xk, p=uk)
        xk = np.array(res["xf"]).reshape(-1)

        # Optional clipping of bounded states after integration
        xk[2] = np.clip(xk[2], model.psi_min, model.psi_max)
        xk[3] = np.clip(xk[3], model.v_min, model.v_max)

        states.append(xk.copy())

    return np.array(states), np.array(inputs), model


def print_short_explanation(states, inputs, model):
    x_final = states[-1, 0]
    y_final = states[-1, 1]
    psi_final = states[-1, 2]
    v_initial = states[0, 3]
    v_mid = states[len(states) // 2, 3]
    v_final = states[-1, 3]

    print("\nShort system behavior explanation:")
    print(
        "For the first 50 steps, the vehicle receives a small positive acceleration "
        "and a very small positive steering input, so it speeds up slightly and drifts "
        "smoothly toward larger y-values."
    )
    print(
        "For the last 50 steps, both inputs switch sign, so the vehicle starts to "
        "decelerate slightly and its curvature changes in the opposite direction."
    )
    print(
        "Because the steering angle magnitude is very small, the heading and lateral "
        "position evolve smoothly without abrupt turns."
    )
    print(
        f"Initial speed: {v_initial:.3f} m/s, speed near midpoint: {v_mid:.3f} m/s, "
        f"final speed: {v_final:.3f} m/s."
    )
    print(
        f"Final position: x = {x_final:.3f} m, y = {y_final:.3f} m, psi = {psi_final:.6f} rad."
    )


if __name__ == "__main__":
    states, inputs, model = simulate_model_test()

    print("Initial state:", states[0])
    print("Final state:  ", states[-1])

    plot_trajectory(states, model, save_path="trajectory.png", show=True)
    plot_states_and_inputs(states, inputs, model.dt, save_path="states_inputs.png", show=True)

    print_short_explanation(states, inputs, model)