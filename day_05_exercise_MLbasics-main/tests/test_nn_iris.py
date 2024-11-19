"""Test the python functions from ./src/nn_iris_solution."""

import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split

sys.path.insert(0, "./src/")

from src.nn_iris import (
    accuracy_most_frequent,
    accuracy_stratified,
    compute_accuracy,
    cv_knearest_classifier,
)


def test_compute_accuracy():
    """Test the compute_accuracy function from the iris solution."""
    y = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    acc = compute_accuracy(y, y_pred)
    assert np.allclose(acc, 0.8)


def test_accuracy_most_frequent():
    """Ensure the accuracy_most_frequent function works as expected."""
    ytrain = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1])
    ytest = np.array([1, 0, 1, 1, 1])
    acc_mf = accuracy_most_frequent(ytrain, ytest)
    assert np.allclose(acc_mf, 0.8)


def test_accuracy_stratified():
    """Test the accuracy_stratified function."""
    np.random.seed(42)  # Set the random seed for reproducibility
    ytrain = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1])
    ytest = np.array([1, 0, 1, 1, 1])
    acc_strat = accuracy_stratified(ytrain, ytest)
    assert np.allclose(acc_strat, 0.4)


def test_cv_knearest_classifier():
    """Test cv_knearest_classifier."""
    # Create a dummy dataset
    in_x, y = make_classification(n_samples=100, n_features=20, random_state=42)
    xtrain, xtest, ytrain, ytest = train_test_split(
        in_x, y, train_size=0.75, random_state=29
    )

    # Call the function
    knn_cv = cv_knearest_classifier(xtrain, ytrain)

    # Check if the returned object is of the expected type
    assert isinstance(knn_cv, GridSearchCV)

    # Get the best score
    best_score = knn_cv.best_score_
    # Get the best score
    best_params = knn_cv.best_params_["n_neighbors"]

    # Perform assertions on the best results
    assert np.allclose(best_score, 0.893333333)
    assert np.allclose(best_params, 4)
