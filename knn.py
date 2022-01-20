import numpy as np
from collections import Counter


class KNN:
    """K Nearest Neighbour Implementation"""

    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        y_prediction = np.asarray([self._predictOne(x) for x in X_test])
        return y_prediction

    def _predictOne(self, x):
        # calculate euclidean distance between the unknown data and all other data in datasets
        distances = np.asarray(
            [self.euclideanDistance(x, x_train) for x_train in self.X_train])
        # sort according to the distance and select the index of first k neighbours
        sortedKIndex = np.argsort(distances)[:self.k]
        # get the y label for the first k neighbours
        labelOfKNeigbours = [self.Y_train[i] for i in sortedKIndex]
        # return the result as per the majority label
        majority = Counter(labelOfKNeigbours).most_common(1)
        return majority[0][0]

    def euclideanDistance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
