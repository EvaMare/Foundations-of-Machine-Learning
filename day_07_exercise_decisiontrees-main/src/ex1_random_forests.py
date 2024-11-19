"""Compare decision trees to random forests in regression."""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn import ensemble, metrics, tree
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def train_dt_and_rf(
    x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> dict[str, list]:
    """Train decision trees and random forests for regression.

    Define a decision tree regressor with max_depth parameter value of 1.
    Train it and compute the MSE on the test set. Repeat these steps for all max_depth
    values between 2 and 30 (including 30). Afterwards, perform all these operations
    with a random forest regressor.

    Args:
        x_train (np.ndarray): The training data.
        x_test (np.ndarray): The test data.
        y_train (np.ndarray): The training labels.
        y_test (np.ndarray): The test labels.

    Returns:
        dict[str, list]: Two lists containing the MSEs for the decision tree and the random forest regressors.
    """
    # 3. create and fit a DT regressor
    # TODO: add your solution here

    # 4. make predictions on test data and compute and print MSE
    # TODO: add your solution here

    # 5. repeat steps 3 and 4 for higher max_depth values
    # TODO: add your solution here

    # 6. repeat steps 3,4 and 5 with random forests
    # TODO: add your solution here

    # 7. return both MSE lists
    # TODO: replace the lists with the correct data
    return {
        "decision_trees_mse": [],  # list with MSEs of decision trees
        "random_forests_mse": [],  # list with MSEs of random forests
    }


if __name__ == "__main__":
    np.random.seed(0)
    # 1. import dataset
    # TODO: add your solution here

    # 2. split dataset into train and test
    # TODO: add your solution here

    # 8. call `train_dt_and_rf` function and get MSE curves
    # TODO: add your solution here

    # 9. plot both MSE curves
    # TODO: add your solution here

    # 10. Observations:
    # TODO: add your observations here
