# Decision Trees vs. Random Forests Exercise

Let's verify that the technique called *bagging* improves decision trees when they become random forests. 

### Task 1: Regression with Random Forests

In this exercise we are looking at the California Housing dataset. Here, each data point consists of 8 census attributes (e.g. average income, building size, location, etc.) together with the median house value.

We will examine at how random forests can be used for regularization and compare them to decision trees. Open the file `src/ex1_random_forests.py` and go to the `__main__` function.

1. Import the California Housing dataset (`sklearn.datasets.fetch_california_housing`).

2. Split the dataset into train and tests sets. Use a test size of 10% and `random_state=21`.

We will now implement the `train_dt_and_rf` function to train decision trees and random forests by following these steps:

3. Create a decision tree regressor (`sklearn.tree.DecisionTreeRegressor`) with a maximum tree depth of 1 and fit it to your training data.
4. Make the predictions of your regressor on the test data and calculate the mean squared error between the predictions and the ground truth targets. For convenience you can use `sklearn.metrics.mean_squared_error`. Print the result.
5. Repeat steps 3 and 4 for a maximum depth of 2, 3, ..., 30. Save the resulting MSEs in an array.
6. Repeat steps 3-5, but this time using a random forest regressor (`sklearn.ensemble.RandomForestRegressor`). Reduce the number of estimators the random forest uses to 10 to speed up the training.
7. Make sure your function returns the two lists of MSEs in a dictionary.

Take a look at your results and analyse them:

8. Call your `train_dt_and_rf` function and get both the MSE curve of the decision trees from step 5 and the MSE curve of the random forests from step 6.
8. Plot both MSE curves together in one figure (x-axis: maximum depth, y-axis: MSE).
9. Look at the curve of the decision trees and how the MSE changes as the maximum depth increases. What do you observe? Why do you think this is happening? How does the curve of the random forests differ from the previous one? Why is this the case?


### Task 2: Visualising Decision Boundaries for Classification

For 2D data it is possible to directly visualize how a classifier divides the plane into classes. Luckily, scikit-learn provides such a ``DecisionBoundaryDisplay`` for its estimators. Like in the previous exercise, we will create decision trees and random forests and compare their performance on some synthetic datasets.

This time, the datasets will be provided by parameters of some functions. In the file the file `src/ex2_decision_boundaries.py` Implement the ``train_and_visualize_decision_boundaries()`` function by following these steps:

1. Create a scatter plot of the dataset provided via the function parameters. Colorize the points according to their class membership (`c=targets`).
2. Fit a decision tree classifier on the whole dataset using the `sklearn.tree` module and plot the tree. Look at the `sklearn.tree` module for help.
3. Create a ``DecisionBoundaryDisplay`` using the ``sklearn.inspection.DecisionBoundaryDisplay.from_estimator`` function and use `vmax=2/0.29` and `cmap=plt.cm.tab10`. To show the data points in the same plot, you can call the ``ax_.scatter()`` method of the display you created and use it like ``plt.scatter()`` before you call ``plt.show()``. In `ax_.scatter()` set `vmax=2/0.29` and `cmap=plt.cm.tab10` as well. This way, all the plots should use the same colors.
4. If you run the script with `python ./src/ex2_decision_boundaries.py`, you will see that your function will be called with five different datasets: vertical lines, diagonal lines, nested circles, half-moons and spirals. Do the decision trees created by the datasets have different complexities? If yes, why do you think is that the case?
5. Now train a random forest classifier from `sklearn.ensemble` on the whole data.
6. Repeat step 3 using the classifier from step 5. How do the decision boundaries of the random forest classifier differ from the ones described by the decision tree classifier?
7. Make sure, your function returns the decision tree classifier and the random forest classifier you created - again using a dictionary.
8. Play around with different values for `n_samples` and `noise` in `make_circles` and `make_moons`.


