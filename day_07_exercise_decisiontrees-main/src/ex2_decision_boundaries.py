"""Compare decision trees to random forests in classification and plot the decision boundaries of the classifiers."""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn import ensemble, inspection, tree
from sklearn.datasets import make_moons


def train_and_visualize_decision_boundaries(
    data: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """Train decision trees and random forests for classification and visualize decision boundaries.

    Create a scatter plot to visualize the data. Train a decision tree and plot
    the tree of the trained model. Plot the decision boundaries of the classifier.
    Then train a random forest and plot the decision boundaries of the new classifier.

    Args:
        data (np.ndarray): Array of data.
        targets (np.ndarray): Array containing the labels.

    Returns:
        dict: Instances of the decision tree classifier and the random forest classifier.
    """
    # 1. create scatter plot
    # TODO: add your solution here

    # 2. fit decision tree classifier and plot resulting tree
    # TODO: add your solution here

    # 3. plot decision boundaries
    # TODO: add your solution here

    # 4. why different complexities?
    # TODO: add your observations here

    # 5. create random forest classifier
    # TODO: add your solution here

    # 6. plot its decision boundaries
    # TODO: add your solution here

    # 7. return the classifiers
    # TODO: replace the lists with the correct data

    return {
        "decision_tree": [],  # the decision tree classifier
        "random_forest": [],  # the random forest classifier
    }


def make_lines(
    n_samples: int,
    n_lines: int,
    line_distance: float,
    x_noise: float,
    angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``n_lines`` with ``n_samples`` each.

    The lines are ``line_distance`` units apart from each other and are rotated ``angle``
    (radians) w.r.t. to the x-axis. Normally distributed noise with standard deviation
    ``x_noise`` is the x-coordinate of each sample.

    Returns:
        tuple (np.ndarray, np.ndarray): Tuple with data of shape (300, 2) and labels of shape (300,).
    """
    line_samples = np.random.random(n_lines * n_samples)

    targets = np.repeat(np.arange(n_lines), repeats=n_samples)
    data = np.tile(line_samples[:, None], reps=(1, 2))
    data[:, 0] *= np.cos(angle)
    data[:, 1] *= np.sin(angle)

    data[:, 0] += targets * line_distance
    data[:, 0] += np.random.normal(size=len(data), scale=x_noise)

    return data * 10, targets


def make_circles(
    n_samples: int,
    radii: List,
    noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate n circles, where n is the length of the array ``radii``, with ``n_samples`` each.

    The i-th circle has radius ``radii[i]``. Normally distributed random noise with standard
    deviation ``noise`` is added to each sample.

    Returns:
        tuple (np.ndarray, np.ndarray): Containing data of shape (1500, 2) and targets of shape (1500,).
    """
    angle_samples = np.random.random(n_samples * len(radii)) * 2 * np.pi
    targets = np.repeat(np.arange(len(radii)), repeats=n_samples)

    data = np.stack([np.cos(angle_samples), np.sin(angle_samples)], axis=1)
    data *= np.repeat(radii, repeats=n_samples)[:, None]
    data += np.random.normal(scale=noise, size=data.shape)

    return data, targets


def make_spiral(
    n_samples: int,
    radii: List,
    noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate n spirals, where n is the length of the array ``radii``, with ``n_samples`` each.

    The i-th spiral has radius ``radii[i]``. Normally distributed random noise with standard
    deviation ``noise`` is added to each sample.

    Returns:
        tuple (np.ndarray, np.ndarray): Containing data and targets.
    """
    noise *= 1000
    angle_samples = np.random.random(n_samples * len(radii)) * 20
    r = angle_samples**2
    targets = np.repeat(np.arange(len(radii)), repeats=n_samples)

    data = np.stack([r * np.cos(angle_samples), r * np.sin(angle_samples) + 3], axis=1)
    data *= np.repeat(radii, repeats=n_samples)[:, None]
    data += np.random.normal(scale=noise, size=data.shape)

    return data, targets


if __name__ == "__main__":
    np.random.seed(0)

    # vertical lines
    data, targets = make_lines(
        n_samples=100,
        n_lines=3,
        line_distance=0.15,
        x_noise=0.01,
        angle=np.pi / 2,
    )
    train_and_visualize_decision_boundaries(data, targets)

    # diagonal lines
    data, targets = make_lines(
        n_samples=100,
        n_lines=3,
        line_distance=0.15,
        x_noise=0.01,
        angle=np.pi / 4,
    )
    train_and_visualize_decision_boundaries(data, targets)

    # circles
    data, targets = make_circles(n_samples=500, radii=[1.8, 2.2, 2.6], noise=0.015)
    train_and_visualize_decision_boundaries(data, targets)

    # half moons
    data, targets = make_moons(noise=0.3, random_state=0)
    train_and_visualize_decision_boundaries(data, targets)

    # spirals
    # use n_samples in [100, 1000], noise in [0., 0.7]
    data, targets = make_spiral(n_samples=300, radii=[-10.0, 10.0], noise=0.25)
    train_and_visualize_decision_boundaries(data, targets)
