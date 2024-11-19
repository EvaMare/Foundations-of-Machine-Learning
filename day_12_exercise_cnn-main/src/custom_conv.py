"""This module ships a function."""
import jax
import jax.numpy as jnp


def my_conv_direct(image: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """Evaluate a selfmade convolution function.

    Implements slow summation in a for loop.

    Args:
        image (jnp.ndarray): The input image of shape [height, width.]
        kernel (jnp.ndarray): A 2d-convolution kernel.

    Return:
        jnp.ndarray: An array with the cross correlation per pixel.

    """
    image_rows, image_cols = image.shape
    kernel_rows, kernel_cols = kernel.shape
    corr = []
    for img_row in range(image_rows - kernel_rows + 1):
        corr_row = []
        for img_col in range(image_cols - kernel_cols + 1):
            # TODO: Compute the convolution here.
            res = 0.0
            corr_row.append(res)
        corr.append(jnp.stack(corr_row))
    return jnp.stack(corr)


# @jax.jit
def get_indices(image: jnp.ndarray, kernel: jnp.ndarray) -> tuple:
    """Optional: Get the indices to set up pixel vectors for convolution by matrix-multiplication.

    Args:
        image (jnp.ndarray): The input image of shape [height, width.]
        kernel (jnp.ndarray): A 2d-convolution kernel.

    Returns:
        tuple: An integer array with the indices, the number of rows in the result,
        and the number of columns in the result.
    """
    # TODO: Implement me.
    idx_list = None
    corr_rows = None
    corr_cols = None
    return idx_list, corr_rows, corr_cols


def my_conv(image: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """Optional: Evaluate an efficient selfmade convolution function.

    Implement `get_indices` to make this work.

    Args:
        image (jnp.ndarray): The input image of shape [height, width.]
        kernel (jnp.ndarray): A 2d-convolution kernel.

    Return:
        jnp.ndarray: An array with the cross correlation per pixel.
    """
    idx_list, corr_rows, corr_cols = get_indices(image, kernel)
    img_vecs = image.flatten()[idx_list]
    corr_flat = img_vecs @ kernel.flatten()
    corr = corr_flat.reshape(corr_rows, corr_cols)
    return jnp.stack(corr)
