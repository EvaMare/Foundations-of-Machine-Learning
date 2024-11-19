"""The nearest neighbour regression."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def gs_knearest_regressor(
    xtrain: np.ndarray, ytrain: np.ndarray
) -> Tuple[KNeighborsRegressor, np.signedinteger, np.ndarray]:
    """Implement the grid search for k-nearest neighbors regressor.

    Split the train set into a train and a validation set (80:20) and train the new train set for k = 1 to 40.
    For each estimator calculate mean squared error on the val set and store the results in an array.
    Determine, which value of k should we take and train the regressor on the whole train set with
    the newly determined value of k.

    Args:
        xtrain (np.ndarray): The training data.
        ytrain (np.ndarray): The training labels.

    Returns:
        (Tuple[KNeighborsRegressor, signedinteger, np.ndarray]):
        KNeighborsRegressor: The model that was trained with the optimal number of k.
        signedinteger: k that produced the smallest mean squared error on the val set.
        np.ndarray: The array with means squared errors for k values between 1 and 39.
    """
    # TODO: Implement me.
    return None


if __name__ == "__main__":
    # as_frame=True loads data in dataframe format, with other metadata besides it
    california_housing = fetch_california_housing(as_frame=True)

    # select only dataframe part
    df = california_housing.frame

    # assign MedHouseVal to y as labels and all other columns to x as features
    # TODO

    # use 'describe' and 'print' to check if we have differences in feature measurements
    # TODO

    # We see that the features have differences in the mean and the standard deviation. We are using an algorithm
    # based on a distance and distance-based algorithms suffer greatly from data that isn't
    # on the same scale, such as this data. The scale of the points will distort the real
    # distance between values. That means that we will need to scale the features on the train data.

    # split the data into test and train set, 80:20, use random_state = 29
    # TODO
    x_train, x_test, y_train, y_test = None

    # print dimensions of test and train data
    # TODO

    # use 'StandardScaler' on train data
    scaler = StandardScaler()
    scaler.fit(x_train)

    # scale both train and test data
    # TODO

    # we can now look at the scaled data
    col_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    scaled_df = pd.DataFrame(x_train, columns=col_names)
    print(scaled_df.describe().T)

    # define a k-nn regressor with k = 1
    # TODO

    # train the regressor and calculate mean squared error on the test set
    # TODO

    # print mean squared error
    # TODO

    # Implement the function 'gs_knearest_regressor' to perform the grid search to
    # determine the best value for k for k-nearest neighbors regressor.
    # Use this function to get the best estimator, the best number for k and
    # the error array of the search. Print, which value of k should we take?
    # TODO
    knn_gs, num_neighbors, error_array = None

    # compute and print MSE of best estimator on test set
    # TODO

    # plot mean squared error for k values between 1 and 39
    plt.figure(figsize=(12, 6))
    plt.plot(
        range(1, 40),
        error_array,
        color="red",
        linestyle="dashed",
        marker="o",
        markerfacecolor="blue",
        markersize=10,
        label="validation error",
    )
    plt.title("K Value MSE")
    plt.xlabel("K Value")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.show()
