"""Test decision boundaries methods."""
import numpy as np
import sklearn.tree

from src.ex2_decision_boundaries import (
    make_lines,
    train_and_visualize_decision_boundaries,
)


def test_train_and_visualize_decision_boundaries():
    """Test if the returned DT and RF classifiers are correct."""
    data, targets = make_lines(
        n_samples=100, n_lines=3, line_distance=0.15, x_noise=0.01, angle=np.pi / 2
    )
    result_dict = train_and_visualize_decision_boundaries(data, targets)
    assert isinstance(result_dict["decision_tree"], sklearn.tree.DecisionTreeClassifier)
    assert isinstance(
        result_dict["random_forest"], sklearn.ensemble.RandomForestClassifier
    )
