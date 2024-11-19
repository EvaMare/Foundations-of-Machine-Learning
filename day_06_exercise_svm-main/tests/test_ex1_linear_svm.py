"""Test linear svm functions."""
import sys

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

sys.path.insert(0, "./src/")

from src.ex1_linear_svm import train_test_svc


def test_train_test_svc():
    """Test the linear SVM classifier."""
    # create dummy data
    x_train = np.array([1, 3, 5, 7, 9]).reshape(-1, 1)
    y_train = np.array([0, 0, 1, 1, 1])
    x_test = np.array([2, 4, 8]).reshape(-1, 1)
    y_test = np.array([0, 0, 1])

    model, accuracy = train_test_svc(x_train, x_test, y_train, y_test)

    # check if returned object is of expected type
    assert isinstance(model, svm.LinearSVC)

    # ensure accuracy matches accuracy_score
    y_pred = model.predict(x_test)
    expected_accuracy = (accuracy_score(y_test, y_pred) * 100).round(1)
    assert np.allclose(accuracy, expected_accuracy)
    assert np.allclose(accuracy, 66.7)
