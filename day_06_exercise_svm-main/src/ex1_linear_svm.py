"""Use linear SVM for iris classification."""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def train_test_svc(
    xtrain: np.ndarray, xtest: np.ndarray, ytrain: np.ndarray, ytest: np.ndarray
) -> Tuple[LinearSVC, float]:
    """Train a linear SVM classifier and compute the accuracy on the test set.

    Define a linear SVM classifier. Train it and compute the accuracy on the test set.
    Express the accuracy as a percentage and round to one decimal place.

    Args:
        xtrain (np.ndarray): The training data.
        xtest (np.ndarray): The test data.
        ytrain (np.ndarray): The training labels.
        ytest (np.ndarray): The test labels.

    Returns:
        Tuple[LinearSVC, float]: The trained linear SVM classifier and the accuracy of the model on the test set.
    """
    # 4.1. create and train linear SVM
    # TODO

    # 4.2. predict on test set and calculate accuracy
    # TODO
    return LinearSVC(), float()


if __name__ == "__main__":
    # 1. load the Iris dataset
    # TODO

    # 2. get access to data, labels and class names
    # TODO

    # 3. split data into training and test sets
    # TODO

    # 4. use function train_test_svc to train a linear SVM model and calculate accuracy on test set
    # TODO
    # 5. print accuracy
    # TODO

    # 6. plot confusion matrix
    # TODO
    pass
