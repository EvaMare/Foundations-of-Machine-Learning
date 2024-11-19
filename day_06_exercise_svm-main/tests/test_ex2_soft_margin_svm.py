"""Test soft margin svm functions."""
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, "./src/")

from src.ex2_soft_margin_svm import cv_svm


def test_cv_svm():
    """Test the cross-validated soft margin SVM classifier."""
    # create dummy dataset
    x, y = make_classification(
        n_samples=100, n_features=20, n_classes=2, random_state=42
    )
    # reshape data to fit into SVM
    x = x.reshape(-1, 20)
    # call function that is to be tested
    clf = cv_svm(x, y)

    # check if returned object is of expected type
    assert isinstance(clf, GridSearchCV)

    # Get the best score
    best_score = clf.best_score_
    # Get the best parameters
    best_param_c = clf.best_params_["C"]

    # Perform assertions on the best results
    assert np.allclose(best_score, 0.99)
    assert np.allclose(best_param_c, 1)
