"""Regularization proof of concept."""
import matplotlib.pyplot as plt
import numpy as np
import pandas


def set_up_point_matrix(axis_x: np.ndarray, degree: int) -> tuple:
    """Set up a point matrix to fit a polynomial.

    The matrix should have to following form:
    [a_1**0       a_1**1      ...  a_1**(degree-1)
     a_2**0       a_2**1      ...  a_2**(degree-1)
     a_3**0       a_3**1      ...  a_3**(degree-1)
     ...          ...         ...  ...
     a_points**0  a_points**1 ...  a_points**(degree-1)]

    Where the entries in the matrix are from the axis_x vector.

    Args:
        axis_x (np.ndarray): The values of the time or x-axis.
        degree (int): The degree of the polynomial.

    Returns:
        tuple: The polynomial point matrix A.
    """
    mat_a = np.zeros((len(axis_x), degree))
    # Fill in powers of axis_x
    for i in range(1, degree):
        mat_a[:, i] = axis_x**i

    return mat_a


if __name__ == "__main__":
    b_noise_p = pandas.read_csv("./data/noisy_signal.tab", header=None)
    b_noise = np.asarray(b_noise_p)

    x_axis = np.linspace(0, 1, num=len(b_noise))

    plt.plot(x_axis, b_noise, ".", label="b_noise")
    plt.show()

# Set the degree of the polynomial
degree = 300  # First degree polynomial (a line), you can adjust this

# Step 1: Set up the point matrix (A_m)
mat_a = set_up_point_matrix(x_axis, degree)

# Step 2: Solve for the coefficients using the normal equation: (A^T * A) * x = A^T * b
A_T_A = np.dot(mat_a.T, mat_a)
A_T_b = np.dot(mat_a.T, b_noise)
    
# Solve for the coefficients
coefficients = np.linalg.solve(A_T_A, A_T_b)

# Step 3: Evaluate the polynomial using the coefficients
y_poly = np.dot(mat_a, coefficients)

# Plot the noisy signal and the fitted polynomial
plt.plot(x_axis, b_noise, ".", label="Noisy Signal")
plt.plot(x_axis, y_poly, label=f"Fitted Polynomial (degree {degree})", color="red")
plt.xlabel("x-axis")
plt.ylabel("b_noise")
plt.title("Noisy Signal and Polynomial Fit")
plt.legend()
plt.show()