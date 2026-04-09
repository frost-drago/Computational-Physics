import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.patches as patches
import os

N_POINTS = 41
DOMAIN_SIZE = 1.0
N_ITERATIONS = 500
TIME_STEP_LENGTH = 0.001
KINEMATIC_VISCOSITY = 0.1
DENSITY = 1.0
HORIZONTAL_VELOCITY_TOP = 1.0

N_PRESSURE_POISSON_ITERATIONS = 50
STABILITY_SAFETY_FACTOR = 0.5

OBSTACLE_X = 0.40      # left edge of obstacle
OBSTACLE_Y = 0.40      # bottom edge of obstacle
OBSTACLE_WIDTH = 0.20
OBSTACLE_HEIGHT = 0.20

SAVE_EVERY = 10   # save one frame every 10 iterations

def main():
    element_length = DOMAIN_SIZE / (N_POINTS - 1)
    x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    X, Y = np.meshgrid(x, y)

    u_prev = np.zeros_like(X)
    v_prev = np.zeros_like(X)
    p_prev = np.zeros_like(X)

    solid = np.zeros_like(X, dtype=bool)

    # obstacle edges in physical coordinates
    x0 = OBSTACLE_X
    y0 = OBSTACLE_Y
    x1 = OBSTACLE_X + OBSTACLE_WIDTH
    y1 = OBSTACLE_Y + OBSTACLE_HEIGHT

    # convert physical coordinates to nearest grid indices
    i0 = np.argmin(np.abs(x - x0))
    i1 = np.argmin(np.abs(x - x1))
    j0 = np.argmin(np.abs(y - y0))
    j1 = np.argmin(np.abs(y - y1))

    solid[j0:j1+1, i0:i1+1] = True

    # actual plotted box dimensions snapped to grid
    x0 = x[i0]
    y0 = y[j0]
    width = x[i1] - x[i0]
    height = y[j1] - y[j0]

    def apply_solid_bc(u, v):
        u[solid] = 0.0
        v[solid] = 0.0
        return u, v

    def central_difference_x(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[1:-1, 2:] - f[1:-1, 0:-2]
        ) / (2 * element_length)
        return diff

    def central_difference_y(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[2:, 1:-1] - f[0:-2, 1:-1]
        ) / (2 * element_length)
        return diff

    def laplace(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[1:-1, 0:-2]
            + f[0:-2, 1:-1]
            - 4 * f[1:-1, 1:-1]
            + f[1:-1, 2:]
            + f[2:, 1:-1]
        ) / (element_length**2)
        return diff

    maximum_possible_time_step_length = 0.5 * element_length**2 / KINEMATIC_VISCOSITY
    if TIME_STEP_LENGTH > STABILITY_SAFETY_FACTOR * maximum_possible_time_step_length:
        raise RuntimeError("Stability is not guaranteed")

    os.makedirs("frames", exist_ok=True)

    # save initial frame at t=0
    save_frame(X, Y, p_prev, u_prev, v_prev, x0, y0, width, height, 0, 0.0)

    for iteration in tqdm(range(1, N_ITERATIONS + 1)):
        d_u_prev__d_x = central_difference_x(u_prev)
        d_u_prev__d_y = central_difference_y(u_prev)
        d_v_prev__d_x = central_difference_x(v_prev)
        d_v_prev__d_y = central_difference_y(v_prev)
        laplace__u_prev = laplace(u_prev)
        laplace__v_prev = laplace(v_prev)

        u_tent = (
            u_prev
            + TIME_STEP_LENGTH * (
                -(u_prev * d_u_prev__d_x + v_prev * d_u_prev__d_y)
                + KINEMATIC_VISCOSITY * laplace__u_prev
            )
        )

        v_tent = (
            v_prev
            + TIME_STEP_LENGTH * (
                -(u_prev * d_v_prev__d_x + v_prev * d_v_prev__d_y)
                + KINEMATIC_VISCOSITY * laplace__v_prev
            )
        )

        # wall BC
        u_tent[0, :] = 0.0
        u_tent[:, 0] = 0.0
        u_tent[:, -1] = 0.0
        u_tent[-1, :] = HORIZONTAL_VELOCITY_TOP

        v_tent[0, :] = 0.0
        v_tent[:, 0] = 0.0
        v_tent[:, -1] = 0.0
        v_tent[-1, :] = 0.0

        u_tent, v_tent = apply_solid_bc(u_tent, v_tent)

        d_u_tent__d_x = central_difference_x(u_tent)
        d_v_tent__d_y = central_difference_y(v_tent)

        rhs = DENSITY / TIME_STEP_LENGTH * (d_u_tent__d_x + d_v_tent__d_y)

        for _ in range(N_PRESSURE_POISSON_ITERATIONS):
            p_next = np.zeros_like(p_prev)
            p_next[1:-1, 1:-1] = 0.25 * (
                p_prev[1:-1, 0:-2]
                + p_prev[0:-2, 1:-1]
                + p_prev[1:-1, 2:]
                + p_prev[2:, 1:-1]
                - element_length**2 * rhs[1:-1, 1:-1]
            )

            # pressure BC
            p_next[:, -1] = p_next[:, -2]
            p_next[0, :] = p_next[1, :]
            p_next[:, 0] = p_next[:, 1]
            p_next[-1, :] = 0.0

            p_next[solid] = p_prev[solid]
            p_prev = p_next

        d_p_next__d_x = central_difference_x(p_next)
        d_p_next__d_y = central_difference_y(p_next)

        u_next = u_tent - TIME_STEP_LENGTH / DENSITY * d_p_next__d_x
        v_next = v_tent - TIME_STEP_LENGTH / DENSITY * d_p_next__d_y

        u_next, v_next = apply_solid_bc(u_next, v_next)

        # wall BC again
        u_next[0, :] = 0.0
        u_next[:, 0] = 0.0
        u_next[:, -1] = 0.0
        u_next[-1, :] = HORIZONTAL_VELOCITY_TOP

        v_next[0, :] = 0.0
        v_next[:, 0] = 0.0
        v_next[:, -1] = 0.0
        v_next[-1, :] = 0.0

        # save frame
        if iteration % SAVE_EVERY == 0 or iteration == N_ITERATIONS:
            current_time = iteration * TIME_STEP_LENGTH
            save_frame(X, Y, p_next, u_next, v_next, x0, y0, width, height, iteration, current_time)

        u_prev = u_next
        v_prev = v_next
        p_prev = p_next


def save_frame(X, Y, p, u, v, x0, y0, width, height, iteration, current_time):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 6))

    speed = np.sqrt(u**2 + v**2)

    contour = ax.contourf(
        X[::2, ::2],
        Y[::2, ::2],
        speed[::2, ::2],
        cmap="coolwarm",
        levels=np.linspace(0, HORIZONTAL_VELOCITY_TOP, 50)
    )
    fig.colorbar(contour, ax=ax)

    ax.streamplot(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2], color="black")

    rect = patches.Rectangle(
        (x0, y0),
        width,
        height,
        linewidth=2,
        edgecolor='black',
        facecolor='gray',
        alpha=0.6,
        zorder=10,
    )
    ax.add_patch(rect)

    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_title(f"Iteration = {iteration}, time = {current_time:.3f} s")

    plt.savefig(f"frames/frame_{iteration:04d}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()