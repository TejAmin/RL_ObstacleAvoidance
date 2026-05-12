import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(states, model, save_path=None, show=True):
    states = np.asarray(states, dtype=float)

    x = states[:, 0]
    y = states[:, 1]

    fig, ax = plt.subplots(figsize=(11, 5))

    # ===== Trajectory =====
    ax.plot(x, y, linewidth=2, label="Vehicle trajectory")

    # ===== Highway =====
    ax.axhline(y=0.0, linestyle="--", label="Road boundary")
    ax.axhline(y=model.lane_width, linestyle=":", label="Lane separator")
    ax.axhline(y=2.0, linestyle=":", label="Right lane center")
    ax.axhline(y=model.road_y_max, linestyle="--")

    # ===== Obstacle =====
    # actual obstacle
    obstacle = plt.Circle(
        (model.obs_x, model.obs_y),
        model.obs_r,
        color="black",
        fill=False,
        linewidth=2,
        label="Obstacle"
    )

    # safety margin
    safety = plt.Circle(
        (model.obs_x, model.obs_y),
        model.obs_r + model.obs_margin,
        color="black",
        linestyle="--",
        fill=False,
        label="Obstacle + safety"
    )

    ax.add_patch(obstacle)
    ax.add_patch(safety)

    # ===== Start / End =====
    ax.scatter(states[0, 0], states[0, 1], marker="o", s=60, label="Start")
    ax.scatter(states[-1, 0], states[-1, 1], marker="x", s=60, label="End")

    # ===== Labels =====
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Vehicle trajectory")
    ax.grid(True)
    ax.set_ylim(0, 10)
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_states_and_inputs(states, inputs, dt, save_path=None, show=True):
    states = np.asarray(states, dtype=float)
    inputs = np.asarray(inputs, dtype=float)

    t_x = np.arange(states.shape[0]) * dt
    t_u = np.arange(inputs.shape[0]) * dt

    channels = [
        (t_x, states[:, 0], "x position [m]",       "x position [m]",      False),
        (t_x, states[:, 1], "y position [m]",        "Position y [m]",      False),
        (t_x, states[:, 2], "Heading ψ [rad]",        "Heading ψ [rad]",     False),
        (t_x, states[:, 3] * 3.6, "Velocity [km/h]", "Velocity [km/h]",     False),
        (t_u, inputs[:, 0], "Acceleration a [m/s²]",  "Acceleration [m/s²]", True),
        (t_u, inputs[:, 1], "Steering angle δ_f [rad]", "Steering angle [rad]", True),
    ]

    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 2.5 * len(channels)), sharex=False)

    for ax, (t, data, legend_label, ylabel, use_step) in zip(axes, channels):
        if use_step:
            ax.step(t, data, where="post", label=legend_label)
        else:
            ax.plot(t, data, label=legend_label)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time [s]")
        ax.legend(loc="upper right")
        ax.grid(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
