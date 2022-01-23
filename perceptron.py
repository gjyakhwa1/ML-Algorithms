import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, epoch=1000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.weights = None
        self.bias = 0

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.adjust_weights()

    def adjust_weights(self):
        sample_size, total_features = self.X_train.shape
        # initial weights and bias
        self.weights = np.zeros(total_features)
        self.bias = 0
        # outputs in terms of 0 and 1
        actual_y = np.array([1 if i > 0 else 0 for i in self.Y_train])
        # training the neural network
        for _ in range(self.epoch):
            for idx, x_i in enumerate(self.X_train):
                y_hat = np.dot(x_i, self.weights)+self.bias
                activated_y_hat = self._activation_function(y_hat)
                # calculate gradient
                gradient = self.learning_rate*(actual_y[idx]-activated_y_hat)
                # update parameters
                self.weights += gradient*x_i
                self.bias += gradient

    def predict(self, X_test):
        y_predicted = np.dot(X_test, self.weights)+self.bias
        y_activated = self._activation_function(y_predicted)
        return y_activated

    def _activation_function(self, x):
        return np.where(x >= 0, 1, 0)
