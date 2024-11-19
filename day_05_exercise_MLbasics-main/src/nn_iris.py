"""The script for the iris problem."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier


def compute_accuracy(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the accuracy of the prediction.

    The accuracy is the most common performance measure for classification tasks.
    Is is calculated as the ration of correct predictions to all predictions.

    Args:
        y (np.ndarray): The array with the ground truth labels.
        y_pred (np.ndarray): The array with the predicted labels.

    Returns:
        float: The accuracy of the prediction.
    """
    # TODO: Implement me.
    return None


def accuracy_most_frequent(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Compute the accuracy of the most frequent model.

    Most frequent models always predict the label that occurs the most in the train set.
    They belong to the class of so-called "dummy models", because the prediction is
    independent of the input. Such models usually serve as a baseline if
    no other models are available.

    Args:
        y_train (np.ndarray): The array with the training labels.
        y_test (np.ndarray): The array with the test labels.

    Returns:
        float: The accuracy of the prediction by the most frequent model on the test labels.
    """
    # TODO: Implement me.
    return None


# optional
def accuracy_stratified(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Compute the accuracy of the stratified model.

    A stratified model assigns random labels based on the ratio of the labels in the train set.

    Args:
        y_train (np.ndarray): The array with the training labels.
        y_test (np.ndarray): The array with the test labels.

    Returns:
        float: The accuracy of the prediction by the stratified model on the test labels.
    """
    # TODO: Implement me.
    return None


def cv_knearest_classifier(x_train: np.ndarray, y_train: np.ndarray) -> GridSearchCV:
    """Train and cross-validate a k-nearest neighbors classifier with the grid search.

    Define a k-nearest neighbors classifier. Use the grid search with the grid from 1 to 25
    and a 3-fold cross-validation to find the best value for the hyperparameter k.

    Args:
        x_train (np.ndarray): The training data.
        y_train (np.ndarray): The training labels.

    Returns:
        GridSearchCV: The trained model that was cross-validated with the grid search.
    """
    # create new knn model
    # TODO
    # create dictionary of all values we want to test for n_neighbors
    # TODO
    # use gridsearch and 3-fold cross-validation to test all values for n_neighbors
    # TODO
    # fit model to data
    # TODO
    return None


if __name__ == "__main__":
    # load iris dataset
    iris = None  # TODO

    # print shape of data matrix and number of target entries
    # TODO
    # print names of labels and of features
    # TODO

    # (optional) use classes distribution (histogram) to check if iris dataset is balanced:
    # find out what the next two lines of code do
    temp = pd.Series(iris.target)
    target_str = temp.apply(lambda i: iris.target_names[i])
    # and use 'pandas.Series.hist' function to plot histogram
    # TODO

    # (optional) use pandas 'scatter_matrix' to visualize some trends in data:
    # represent iris as pandas data frame
    # TODO
    # create scatter matrix from dataframe, color by target; plot matrix
    # TODO

    # create train and test split with the ratio 75:25 and print their dimensions
    # TODO

    # create a k-nn, with k = 1
    # TODO

    # train the classifier on the train set
    # TODO

    # predict labels first for train and then for test data
    # TODO

    # implement and use 'compute_accuracy' to evaluate your model by calculating accuracy on train and test set
    # TODO
    # print both accuracies
    # TODO

    # implement and use 'accuracy_most_frequent' to compute accuracy of most frequent model
    # TODO
    # print result
    # TODO

    # (optional) implement and use 'accuracy_stratified' to compute and print accuracy of the stratified model
    # TODO

    # compute confusion matrix for test set
    # TODO

    # (optional) compute and print test set accuracy from confusion matrix
    # TODO

    # plot heatmap of confusion matrix for test set
    # TODO

    # implement and use 'cv_knearest_classifier' to perform cross validation with grid search
    # TODO

    # print top performing n_neighbors value
    # TODO

    # print mean score for the top performing value of n_neighbors
    # TODO

    # get access to best estimator
    # TODO
    # calculate train and test accuracy on best estimator
    # TODO

    # evaluate your model by calculating accuracy on train and test set
    # TODO
    # print both accuracies
    # TODO

    # plot heatmap of the confusion matrix for test set
    # TODO
