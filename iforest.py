import numpy as np
import pandas as pd
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
        for i in range(self.n_trees):
            X_prime = X[np.random.choice(X.shape[0], self.sample_size, replace=False)]
            self.trees.append(IsolationTree(height_limit=int(np.ceil(np.log2(self.sample_size)))).fit(X=X_prime))

        return self

    def c(self, size):
        if size == 2:
            return 1
        if size <= 2:
            return 0
        return 2*(np.log(size-1)+0.5772156649)-2*(size-1)/size

    def single_path_len(self, tree, x_i, e=0): # single tree and single element in X
        if isinstance(tree, exTreeNode):
            return e + self.c(tree.size)
        a = tree.split_att # index of column of X
        if x_i[a] < tree.split_point:
            return self.single_path_len(tree.left, x_i, e+1)
        if x_i[a] >= tree.split_point:
            return self.single_path_len(tree.right, x_i, e+1)


    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """

        avg_lens = np.array([[0]])
        for i in range(X.shape[0]):
            x_i_lens = []
            for t in range(1, self.n_trees):
                x_i_lens.append(self.single_path_len(self.trees[t], X[i]))
            avg_l = sum(x_i_lens)/self.n_trees
            avg_lens = np.append(avg_lens, [[avg_l]], axis=0)
        return avg_lens[1:]


    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        scores = []
        path_X = self.path_length(X)
        for i in range(path_X.shape[0]):
            s = 2**(-1*(path_X[i])/(self.c(self.sample_size)))
            scores.append(s)
        return np.array(scores)



    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        binary_scores = [1 if score >= threshold else 0 for score in scores]
        return binary_scores

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        scores = self.anomaly_score(X)
        predictions = self.predict_from_anomaly_scores(scores=scores, threshold=threshold)
        return predictions


class inTreeNode:
    def __init__(self,  split_point, left=None, right=None, split_att=None, n_nodes=0):
        self.split_point = split_point
        self.left = left
        self.right = right
        self.split_att = split_att
        self.n_nodes = n_nodes
        self.value = (split_att, split_point)

    def __repr__(self):
        return self.value.__repr__()

class exTreeNode:
    def __init__(self, size=None, depth=None):
        self.size = size
        self.depth = depth

    def __repr__(self):
        return self.size.__repr__()


class IsolationTree:
    def __init__(self, height_limit, n_nodes=1):  #n_nodes= 0, e = 0
        self.height_limit = height_limit
        self.n_nodes = n_nodes


    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.
        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        if self.height_limit == 0 or len(X) <= 1:
            self.n_nodes += 1
            return exTreeNode(size=len(X))
        else:
            self.n_nodes += 1
            q = np.random.randint(X.shape[1])
            column = X[:,q]
            p = np.random.uniform(min(column), max(column))
            X_left = X[p>X[:,q]]
            X_right = X[p<=X[:,q]]
            self.root = inTreeNode(split_point=p, split_att= q,
                                left=IsolationTree(height_limit=self.height_limit-1, n_nodes=self.n_nodes).fit(X_left),
                                right=IsolationTree(height_limit=self.height_limit-1, n_nodes=self.n_nodes).fit(X_right))
            self.root.n_nodes = self.n_nodes


        return self.root

# https://stackoverflow.com/questions/13066249/filtering-lines-in-a-numpy-array-according-to-values-in-a-range
# https://stackoverflow.com/questions/14262654/numpy-get-random-set-of-rows-from-2d-array

def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    ...
    threshold = 1.0
    TPR = 0.0
    #binary_scores = [1 if score >= threshold else 0 for score in scores]
    #confusion = confusion_matrix(y_true=y, y_pred=binary_scores)
    #TN, FP, FN, TP = confusion.flat
    #TPR = TP / (TP + FN)
    #FPR = FP / (FP + TN)
    while TPR < desired_TPR and threshold != 0:
        binary_scores = [1 if score >= threshold else 0 for score in scores]
        confusion = confusion_matrix(y_true=y, y_pred=binary_scores)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        threshold -= 0.001


    return threshold, FPR
