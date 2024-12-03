"""Implement 2d gradient descent using jax."""

import jax
import jax.numpy as np
import numpy as nnp

from util import write_movie


def bumpy_function(pos: np.ndarray) -> np.ndarray:
    """Return values from an even bumpier function.

    This even bumpier function is hard to optimize.
    It will require momentum.

    Args:
        pos (np.ndarray): The position array [x, y].

    Returns:
        np.ndarray: The height value z.
    """
    return (
        pos[0] * pos[0]
        + pos[1] * pos[1]
        + np.cos(pos[0] * 2 * np.pi)
        + np.sin(pos[1] * 2 * np.pi)
        + (pos[0] > 0).astype(pos.dtype) * 0.5
        + np.tanh(np.sqrt(pos[0] ** 2 + pos[1] ** 2)) * 10
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    gradient_fn = jax.grad(bumpy_function) 

    nx, ny = (1001, 1001)
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    mx, my = np.meshgrid(x, y)
    pos = np.stack((mx, my))
    mz = bumpy_function(pos)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(mx, my, mz, cmap=cm.coolwarm)
    fig.colorbar(surf)

    plt.show()

    plt.contourf(mx, my, mz)
    plt.colorbar()

    start_pos = np.array((2.9, -2.9))
    step_size = 0.1
    alpha = 0.9
    step_total = 100

    pos_list = [start_pos]
    grad = np.array((0.0, 0.0))
    velocity = np.array((0.0, 0.0)) #momentum term

    for _ in range(step_total):
        current_pos = pos_list[-1]
        grad = gradient_fn(current_pos)  # Compute gradient at current position
        velocity = alpha * velocity - step_size * grad  # Update velocity
        new_pos = current_pos + velocity  # Update position
        pos_list.append(new_pos)

    for pos in pos_list:
        plt.plot(pos[0], pos[1], ".r")
    plt.show()

    write_movie(
        nnp.array(mx),
        nnp.array(my),
        nnp.array(mz),
        pos_list,
        "writer_grad_bumpy_plot_jax",
    )
