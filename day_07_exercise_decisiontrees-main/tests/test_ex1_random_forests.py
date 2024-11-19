"""Test that random forests perform better than decision trees."""
import numpy as np
from sklearn.model_selection import train_test_split

from src.ex1_random_forests import train_dt_and_rf


def test_train_dt_and_rf():
    """Test for correct return value and if the MSEs have the required structure."""
    np.random.seed(2)
    n_samples, n_features = 100, 5
    x = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.1,
        random_state=21,
    )

    result_dict = train_dt_and_rf(x_train, x_test, y_train, y_test)
    dt_mse = result_dict["decision_trees_mse"]
    rf_mse = result_dict["random_forests_mse"]

    assert len(dt_mse) == len(rf_mse), "Returned arrays do not have the same length."
    assert len(dt_mse) == 30, "Returned arrays should each contain 30 values."
    assert np.all(dt_mse[10:] > rf_mse[10:]), (
        "At some point, the MSEs of decision trees should be higher than "
        "the MSEs of random forests."
    )
