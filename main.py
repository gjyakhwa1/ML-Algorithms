from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def accuracy(actual_value, predictions):
    return np.sum(predictions == actual_value)/len(actual_value)


def KNNMain():
    # For K Nearest Neighbours
    from knn import KNN
    if __name__ == "__main__":
        iris = datasets.load_iris()
        X_train, X_test, Y_train, Y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=1000)
        clf = KNN(k=7)
        clf.fit(X_train, Y_train)
        result = clf.predict(X_test)
        print("Accuracy:", accuracy(result, Y_test))


def LinearRegressorMain():
    # For Linear Regression
    from linearRegression import LinearRegressor
    if __name__ == "__main__":
        # Generates a random data for regression problems
        X, Y = datasets.make_regression(
            n_samples=300, n_features=1, noise=20, random_state=58)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=1000)

        clf = LinearRegressor()
        clf.fit(X_train, Y_train)
        predictions = clf.predict(X_test)
        print("Mean Square Error:", clf.mean_square_error(Y_test, predictions))
        y_prediction_line = clf.predict(X)
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(X, Y)
        plt.plot(X, y_prediction_line, color="black",
                 linewidth=2, label="Prediction")
        plt.show()


def LogisticRegressorMain():
    # For Logistic Regression
    from logisticRegression import LogisticRegressor
    # For visualizing confusion matrix

    def visualizer(y_test, predictions):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    if __name__ == "__main__":
        cancer = datasets.load_breast_cancer()
        X, Y = cancer.data, cancer.target

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=1000)

        clf = LogisticRegressor(learning_rate=0.001, epoch=1000)
        clf.fit(X_train, Y_train)
        predictions = clf.predict(X_test)
        print("Accuracy:", accuracy(Y_test, predictions))
        visualizer(Y_test, predictions)


def NaiveBayesMain():
    from naiveBayes import NaiveBayes
    if __name__ == "__main__":
        X, Y = datasets.make_classification(
            n_samples=1000, n_features=10, n_classes=2, random_state=123
        )
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=123
        )

        clf = NaiveBayes()
        clf.fit(X_train, Y_train)
        predictions = clf.predict(X_test)
        print(np.unique(Y))
        print("Naive Bayes classification accuracy:",
              accuracy(Y_test, predictions))


NaiveBayesMain()
