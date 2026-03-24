import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(states, model, save_path=None, show=True):
    states = np.asarray(states, dtype=float)

    x = states[:, 0]
    y = states[:, 1]

    fig, ax = plt.subplots(figsize=(11, 5))

    # Trajectory
    ax.plot(x, y, linewidth=2, label="Vehicle trajectory")

    # Highway boundaries for 2 lanes
    ax.axhline(y=0.0, linestyle="--", label="Road boundary")
    ax.axhline(y=model.lane_width, linestyle=":", label="Lane separator")
    ax.axhline(y=2.0, linestyle=":", label="Right lane center")
    ax.axhline(y=model.road_y_max, linestyle="--")

    # Start and end markers
    ax.scatter(states[0, 0], states[0, 1], marker="o", s=60, label="Start")
    ax.scatter(states[-1, 0], states[-1, 1], marker="x", s=60, label="End")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Vehicle trajectory")
    ax.grid(True)
    ax.set_ylim(0, 8)
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

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # x-position
    axes[0, 0].plot(t_x, states[:, 0])
    axes[0, 0].set_title("x [m]")
    axes[0, 0].grid(True)

    # y-position
    axes[0, 1].plot(t_x, states[:, 1])
    axes[0, 1].set_title("y [m]")
    axes[0, 1].grid(True)

    # heading
    axes[1, 0].plot(t_x, states[:, 2])
    axes[1, 0].set_title("psi [rad]")
    axes[1, 0].grid(True)

    # speed
    axes[1, 1].plot(t_x, states[:, 3])
    axes[1, 1].set_title("v [m/s]")
    axes[1, 1].grid(True)

    # acceleration input
    axes[2, 0].step(t_u, inputs[:, 0], where="post")
    axes[2, 0].set_title("a [m/s²]")
    axes[2, 0].grid(True)

    # steering input
    axes[2, 1].step(t_u, inputs[:, 1], where="post")
    axes[2, 1].set_title("delta_f [rad]")
    axes[2, 1].grid(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
