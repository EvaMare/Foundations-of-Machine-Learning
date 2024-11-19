"""Analyse corona virus case data with support vector machines."""
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


def plot_curve(curve: np.ndarray) -> None:
    """Plot a curve of confirmed coronavirus cases.

    Args:
        curve (np.ndarray): y-axis of the number of covid cases for len(curve) days.
    """
    plt.figure(figsize=(10, 5))
    plt.title("Number of confirmed Coronavirus cases in the world", size=20)
    plt.plot(range(len(curve)), curve)


def plot_prediction(
    raw_data: np.ndarray,
    available_days: int,
    max_value: np.int64,
    recursive_predictions: np.ndarray,
) -> None:
    """Plot the raw data and the predicted values.

    Args:
        raw_data (np.ndarray): The raw data to plot.
        available_days (int): The days to plot.
        max_value (np.int64): The maximum number of covid cases.
        recursive_predictions (np.ndarray): Predictions for the following days.
    """
    day1 = datetime.datetime.strptime("1/22/2020", "%m/%d/%Y")
    day50 = day1 + datetime.timedelta(days=50)
    available_date_range = [
        day50 + datetime.timedelta(days=x) for x in range(available_days)
    ]
    future_date_range = [
        available_date_range[-1] + datetime.timedelta(days=x)
        for x in range(1, len(recursive_predictions) + 1)
    ]
    full_available_date_range = [
        day1 + datetime.timedelta(days=x) for x in range(len(raw_data))
    ]

    plt.figure(figsize=(20, 10))
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.plot(full_available_date_range, raw_data)
    plt.plot(future_date_range, recursive_predictions, linestyle="dashed", color="red")
    plt.legend(["Real values", "Predictions"], prop={"size": 20}, loc=2)
    plt.xticks(rotation=90)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, max_value * 2))

    plt.savefig("./figures/time_series_preds.png")

    print("Number of confirmed cases:")
    for i, day in enumerate(future_date_range):
        print(day.date(), ":", recursive_predictions[i])


def recursive_forecast(
    model: SVR, start_values: np.ndarray, days_to_forecast: int
) -> np.ndarray:
    """Recursivley use new predictions to generate time series predictions for the future.

    Args:
        model (SVR): The trained model for future predictions.
        start_values (np.ndarray): Array of shape (numberOfPreviousData,).
        days_to_forecast (int): Number of days to forecast.

    Returns:
        np.ndarray: Predictions for the next days_to_forecast.
    """
    moving_x = start_values.copy()
    predictions = np.zeros(days_to_forecast)
    for i in range(days_to_forecast):
        mov_res = moving_x.reshape(1, -1)
        new_forecast = 0  # TODO 7.1. predict values for moving x
        for j in range(moving_x.shape[0] - 1):
            moving_x[j] = 0  # TODO 7.2. shift
        moving_x[-1] = new_forecast[0]
        predictions[i] = new_forecast[0]
    return predictions


def cv_svr(train_x: np.ndarray, train_y: np.ndarray) -> GridSearchCV:
    """Find the best parameters for a SVR model with grid search.

    Train and cross-validate a soft margin SVM regressor with the grid search.

    Define an SVM regressor. Use the grid search and a 5-fold cross-validation
    to find the best value for the hyperparameters 'C', 'gamma' and 'epsilon'.

    Args:
        train_x (np.ndarray): The training data.
        train_y (np.ndarray): The training labels.

    Returns:
        GridSearchCV: The trained model that was cross-validated with the grid search.
    """
    # 5.1. define dictionary with parameter grids
    # TODO
    # 5.2. initialize svm regressor and perform grid search
    # TODO

    return None


if __name__ == "__main__":
    # read dataset
    df = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/"
        "master/csse_covid_19_data/csse_covid_19_time_series/"
        "time_series_covid19_confirmed_global.csv"
    )

    # only choose first 82 days
    df = df[df.columns[:82]]

    # compute total number of cases
    first_date_index = df.columns.get_loc("1/22/20")
    df = df[df.columns[first_date_index:]]
    raw_data = df.to_numpy()
    raw_data = np.sum(raw_data, axis=0)

    # only use last part of data
    raw_data_short = raw_data[50:]

    # 1. plot curves using 'plot_curve' function above
    # TODO

    # 2. set number of days you want to forecast
    # and number of days that will be taken into account for forecast
    days_to_forecast = 0  # TODO
    num_previous_data = 0  # TODO

    # 3. build dataset for training and testing
    num_train_test_records = len(raw_data_short) - num_previous_data
    x = np.zeros((num_train_test_records, num_previous_data))
    y = np.zeros(num_train_test_records)
    for i in range(num_train_test_records):
        for j in range(num_previous_data):
            x[i, j] = raw_data_short[0]  # TODO: set training samples.
        y[i] = raw_data_short[0]  # TODO: set test samples.

    # split dataset into train and test sets
    x_train = x.copy()
    y_train = y.copy()
    x_test = raw_data_short[-num_previous_data:]

    # 4. normalize input data to its max value, such that it lies between [0,1]
    # TODO

    # 5. use function 'cv_svr' to perform hyperparameter search with cross validation
    # TODO

    # 6. print parameters found with cross-validation
    # TODO

    # 8. make predictions for next 5 days; round and denormalize predictions
    # TODO

    # 9. use 'plot_prediction' to plot predicted results
    # TODO
