"""Get the computer to find waldo."""
import jax.numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# from custom_conv import my_conv_direct
# from jax.scipy.signal import correlate2d
# from custom_conv import my_conv


if __name__ == "__main__":
    problem_image = np.array(Image.open("./data/waldo/waldo_space.jpg"))
    waldo = np.array(Image.open("./data/waldo/waldo_small.jpg"))

    problem_image = np.mean(problem_image, -1)[1000:1500, 1000:1500]
    waldo = np.mean(waldo, -1)

    plt.imshow(waldo)
    plt.show()

    plt.imshow(problem_image)
    plt.show()

    mean = np.mean(problem_image)
    std = np.std(problem_image)
    problem_image = (problem_image - mean) / std
    waldo = (waldo - mean) / std

    # TODO: Choose which function to use for convolution.

    # Selfmade conv slow
    # conv_res = my_conv_direct(problem_image, waldo)

    # Built in function.
    # conv_res = correlate2d(problem_image, waldo, mode="same", boundary="fill")

    # Selfmade conv fast
    # conv_res = my_conv(problem_image, waldo)

    max = np.argmax(conv_res)
    idx = np.unravel_index(max, conv_res.shape)
    print(idx)
    plt.imshow(np.log(np.abs(conv_res)))
    plt.plot(idx[1], idx[0], "x")
    plt.colorbar()
    plt.show()

    plt.imshow(problem_image)
    plt.show()
