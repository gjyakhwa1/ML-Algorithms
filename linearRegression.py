import numpy as np


class LinearRegressor:
    """Linear Regression for continuous data"""

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
        self.weights = np.zeros(total_features)
        for _ in range(self.epoch):
            y_predicted = np.dot(self.X_train, self.weights)+self.bias
            # computing gradients
            dw = (1/sample_size)*np.dot(self.X_train.T,
                                        (y_predicted-self.Y_train))
            db = (1/sample_size)*np.sum(y_predicted-self.Y_train)
            # updating the parameters
            self.weights -= self.learning_rate*dw
            self.bias -= self.bias*db
        print("Here")

    def predict(self, X_test):
        y_predicted = np.dot(X_test, self.weights)+self.bias
        return y_predicted

    def mean_square_error(self, Y_actual, Y_predicted):
        return np.mean((Y_actual-Y_predicted)**2)
