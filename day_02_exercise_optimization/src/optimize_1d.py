"""Implement gradient descent in 1d."""
import matplotlib.pyplot as plt
import numpy as np


def parabola(x: float) -> float:
    """Square the input to compute x^2."""
    return x * x


def derivative_parabola(x: float) -> float:
    """Compute the derivate of the parabola."""
    return 2 * x


if __name__ == "__main__":
    start_pos = 5.0
    step_size = 0.1  
    step_total = 50 

    pos_list = [start_pos]
    for _ in range(step_total):
        gradient = derivative_parabola(pos_list[-1])
        new_pos = pos_list[-1] - step_size * gradient
        pos_list.append(new_pos)

    x = np.linspace(-5, 5, 100)
    plt.title("Minimize f(x) on parabola")
    plt.plot(x, tuple(parabola(xel) for xel in x))
    plt.plot(pos_list, tuple(parabola(pos) for pos in pos_list), ".")
    plt.show()

    plt.title("f'(x)")
    plt.plot(tuple(derivative_parabola(x) for x in pos_list))
    plt.show()
