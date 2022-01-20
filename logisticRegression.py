import numpy as np
from linearRegression import LinearRegressor


class LogisticRegressor:
    """Logistic Regression """

    def __init__(self, learning_rate=0.01, epoch=1000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.weights = None
        self.bias = 0

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.gradient_descent()

    def gradient_descent(self):
        sample_size, total_features = self.X_train.shape
        # initialize weights
        self.weights = np.zeros(total_features)
        # tuning the parameters
        for _ in range(self.epoch):
            y_value = np.dot(self.X_train, self.weights)+self.bias
            y_predicted = self._sigmoid(y_value)
            # compute gradient
            dw = (1/sample_size)*np.dot(self.X_train.T,
                                        (y_predicted-self.Y_train))
            db = (1/sample_size)*np.sum((y_predicted-self.Y_train))

            # update parameters
            self.weights -= self.learning_rate*dw
            self.bias -= self.learning_rate*db

    def predict(self, X_test):
        y_value = np.dot(X_test, self.weights)+self.bias
        # calculate output in terms of 0 and 1
        y_sigmoid = self._sigmoid(y_value)
        y_predicted = np.asarray([1 if y > 0.5 else 0 for y in y_sigmoid])
        return y_predicted

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))
