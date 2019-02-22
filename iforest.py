import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix

# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.trees = []

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        for i in range(n_trees):
            self.trees.append(IsolationTree.fit(X))

        return self  # WHY?

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."


class inTreeNode:
    def __init__(self,  split_point, left=None, right=None, split_att=None):
        self.split_point = split_point
        self.left = left
        self.right = right
        self.split_att = split_att

class exTreeNode:
    def __init__(self, size=None, depth=None):
        self.size = size
        self.depth = depth


class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.e = 0
        # self.root = None

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.
        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        e = 1
        if self.e >= self.height_limit or len(X) <= 1:
            e += 1
            return exTreeNode(len(X), depth=e)
        else:
            e += 1
            q = np.random.randint(len(X))
            column = sorted(X[:,q])
            p = random.choice(column)
            X_left = X[p>X[:,q]]
            X_right = X[p<=X[:,q]]
            self.root = inTreeNode(split_point=p, split_att=q,
                                left=IsolationTree(self.height_limit-e).fit(X_left),  ## NEEDS TO BE ALL THE COLUMNS
                                right=IsolationTree(self.height_limit-e).fit(X_right))


        return self.root

# https://stackoverflow.com/questions/13066249/filtering-lines-in-a-numpy-array-according-to-values-in-a-range

def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    ## scikit - learn's confusion_matrix() (for use in find_TPR_threshold()).
    ...
