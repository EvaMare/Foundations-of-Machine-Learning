"""Test functions from time series prediction."""
import sys

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

sys.path.insert(0, "./src/")

from src.ex3_time_series import cv_svr, recursive_forecast


def test_recursive_forecast():
    """Test the recursive future time series predictions."""
    # generate some monotonically increasing sample data
    np.random.seed(29)  # set random seed for reproducibility
    num_samples = 50
    train_data = np.linspace(0.1, 1.0, num_samples)  # monotonically increasing values
    train_data = train_data + np.random.normal(
        0, 0.05, num_samples
    )  # adding some noise
    train_data = np.maximum(train_data, 0)  # ensure non-negative values
    train_data = np.cumsum(train_data)  # creating cumulative sum

    num_previous_data = 7

    num_train_test_records = len(train_data) - num_previous_data

    # build dataset for training and testing
    x = np.zeros((num_train_test_records, num_previous_data))
    y = np.zeros(num_train_test_records)
    for i in range(num_train_test_records):
        for j in range(num_previous_data):
            x[i, j] = train_data[i + j]
        y[i] = train_data[i + num_previous_data]

    x_train = x.copy()
    y_train = y.copy()
    # take last 7 values from training data as start_values
    x_test = train_data[-num_previous_data:]

    # normalize data for better SVM training
    max_value = np.amax(train_data)
    x_train = x_train / max_value
    y_train = y_train / max_value
    x_test = x_test / max_value

    # create and fit sample SVR model
    model = SVR(C=1000.0, epsilon=0.001, gamma="scale")
    model.fit(x_train, y_train)

    days_to_forecast = 3
    start_values = x_test
    # call recursive_forecast function
    predictions = recursive_forecast(model, start_values, days_to_forecast) * max_value

    # make sure predictions have the correct shape
    assert predictions.shape == (days_to_forecast,)
    # make sure all predictions are non-negative
    assert np.all(predictions >= 0)
    # check if predictions are monotonically increasing
    for i in range(1, days_to_forecast):
        assert predictions[i] >= predictions[i - 1]


def test_cv_svr():
    """Test the cross-validated soft margin SVM regressor."""
    # create dummy dataset
    np.random.seed(42)  # set random seed for reproducibility
    num_samples = 50
    num_features = 3
    x = np.random.rand(num_samples, num_features)
    y = np.random.rand(num_samples)

    # call function that is to be tested
    model = cv_svr(x, y)

    # check if returned object is of expected type
    assert isinstance(model, GridSearchCV)

    # get best score
    best_score = model.best_score_
    # get best parameters
    best_param_c = model.best_params_["C"]
    best_param_eps = model.best_params_["epsilon"]

    # perform assertions on best results
    assert np.allclose(best_score, -0.0183686)
    assert np.allclose(best_param_c, 0.1)
    assert np.allclose(best_param_eps, 0.1)
