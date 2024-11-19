"""Test the python functions from ./src/nn_regression."""

import sys

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, "./src/")

from src.nn_regression import gs_knearest_regressor


def test_gs_knearest_regressor() -> None:
    """Ensure the `gs_knearest_regressor` function from src.nn_regression works as expected."""
    # Create a dummy dataset
    np.random.seed(2)
    n_samples, n_features = 100, 5
    x_train = np.random.rand(n_samples, n_features)
    y_train = np.random.rand(n_samples)

    # Call the function
    knn, num_neighbors, error_array = gs_knearest_regressor(x_train, y_train)

    # Check the returned objects
    assert isinstance(knn, KNeighborsRegressor)
    assert isinstance(error_array, np.ndarray)
    assert error_array.shape[0] == 39

    # Generate a validation set
    x_val = np.random.rand(10, n_features)
    y_val = np.random.rand(10)

    # Predict on the validation set for the best k value
    y_pred = knn.predict(x_val)
    mse = mean_squared_error(y_val, y_pred)

    # Define expected values for assertions
    expected_k = 16
    expected_mse = 0.05622218362678164

    # Perform assertions for numerical values
    assert num_neighbors == expected_k
    assert np.allclose(mse, expected_mse)
