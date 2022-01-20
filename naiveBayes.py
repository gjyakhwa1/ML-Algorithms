import numpy as np


class NaiveBayes:
    """Naive Bayes For Classification problems"""

    def fit(self, X_train, Y_train):
        sample_size, total_features = X_train.shape
        self.classes = np.unique(Y_train)
        total_classes = len(self.classes)

        # create an array for storing mean,variance and prior probability for each class for each features
        # row contains classes and features in each column
        self.mean = np.zeros((total_classes, total_features), dtype=np.float64)
        self.variance = np.zeros(
            (total_classes, total_features), dtype=np.float64)
        # prior probability for each class
        self.prior = np.zeros(total_classes, dtype=np.float64)

        for index, c in enumerate(self.classes):
            # using booelan indexing to filter selection according to classes
            X_class = X_train[Y_train == c]
            self.mean[index, :] = np.mean(
                X_class, axis=0)  # mean for each features
            self.variance[index, :] = np.var(X_class, axis=0)
            self.prior[index] = (X_class.shape[0])/sample_size

    def predict(self, X_test):
        y_predicted = [self._helper_predict(x) for x in X_test]
        return y_predicted

    def _helper_predict(self, x):
        posteriors_probability = []
        for index, c in enumerate(self.classes):
            prior = np.log(self.prior[index])  # calculate prior probbiliy p(y)
            # log(p(x1|y))+log(p(x2|y))+..+log(p(xn|y))
            posterior = np.sum(np.log(self._normal_distribution(index, x)))
            total_probability = prior+posterior
            posteriors_probability.append(total_probability)
        # select a particular class based on probability
        return self.classes[np.argmax(posteriors_probability)]

    def _normal_distribution(self, index, x):
        mean = self.mean[index]
        variance = self.variance[index]
        exp_calc = (1/2)*((x-mean)/variance)**2
        return (np.exp(-exp_calc))/np.sqrt(2*np.pi*variance)
