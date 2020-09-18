import numpy as np
import pandas as pd
import numpy as np
import scipy
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import MinMaxScaler


random.seed(0)


def newton_equation(m1=1, m2=1, r=1, G=scipy.constants.G):
    """
    Newton´s Equation
    :param m1: Mass first body in kg
    :param m2: Mass second body in kg
    :param r: Distance between m1 and m2 in meters
    :param G: Gravitational constant
    :return: Newtons equation in the Internation System of Units
    """
    return G * m1 * m2 * (1 / r ** 2)


def make_newton(
    samples=100,
    m1_min=0.001,
    m1_max=1,
    m2_min=0.001,
    m2_max=1,
    r_min=1,
    r_max=10,
    G=scipy.constants.G,
    dataframe=True,
):
    """
    Creates a sample of data of the Newton Equation
    :param samples: Number of samples
    :param m1_min: Minimum value in the range of the mass of the first body
    :param m1_max: Maximum value in the range of the mass of the first body
    :param m2_min: Minimum value in the range of the mass of the second body
    :param m2_max: Maximum value in the range of the mass of the second body
    :param r_min: Minimum value of the distance between the bodies
    :param r_max: Maximum value of the distance between the bodies
    :param dataframe: wether it returns a dataframe or an array
    :return: data sample
    """
    data = []
    for n in range(samples):
        m1 = (m1_max - m1_min) * np.random.random() + m1_min
        m2 = (m2_max - m2_min) * np.random.random() + m2_min
        r = (r_max - r_min) * np.random.random() + r_min

        data.append([m1, m2, r, newton_equation(m1=m1, m2=m2, r=r, G=G)])
    if dataframe:
        return pd.DataFrame(data=data, columns=["m1", "m2", "r", "f"])
    else:
        return data


class NewtonRegressor(BaseEstimator, ClassifierMixin):
    def __init__(self, G=scipy.constants.G):
        self.G = G

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        if isinstance(X, pd.DataFrame):
            return (
                self.G
                * X.values[:, 0]
                * X.values[:, 1]
                / (X.values[:, 2] * X.values[:, 2])
            )
        if isinstance(X, np.ndarray):
            return self.G * X[:, 0] * X[:, 1] / (X[:, 2] * X[:, 2])


class NewtonClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5, G=scipy.constants.G):
        self.threshold = threshold
        self.G = G

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y

        self.lower = MinMaxScaler().fit(y[y < self.threshold].reshape(1, -1))
        self.upper = MinMaxScaler().fit(y[y > self.threshold].reshape(1, -1))

        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        threshold = self.threshold

        # Calculate Newton's Eq and classify by threshold
        if isinstance(X, pd.DataFrame):
            eq = (
                self.G
                * X.values[:, 0]
                * X.values[:, 1]
                / (X.values[:, 2] * X.values[:, 2])
            )
            return [1 if a_ > threshold else 0 for a_ in eq]
        if isinstance(X, np.ndarray):
            eq = self.G * X[:, 0] * X[:, 1] / (X[:, 2] * X[:, 2])
            return [1 if a_ > threshold else 0 for a_ in eq]

    def predict_regression(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Calculate Newton's Eq
        if isinstance(X, pd.DataFrame):
            predictions = (
                self.G
                * X.values[:, 0]
                * X.values[:, 1]
                / (X.values[:, 2] * X.values[:, 2])
            )
        if isinstance(X, np.ndarray):
            predictions = self.G * X[:, 0] * X[:, 1] / (X[:, 2] * X[:, 2])

        return [self.transform_scaler(yi) for yi in predictions]

    def transform_scaler(self, yi):
        threshold = self.threshold
        print(self.upper.transform(np.array([5]).reshape(1, -1)))
        if yi > threshold:
            return self.upper.transform(np.array([yi]).reshape(1, -1))
        if yi < threshold:
            return self.lower.transform(np.array([yi]).reshape(1, -1))
        print("done")


if __name__ == "__main__":

    G = 10
    data = make_newton(samples=1_000, G=G)
    test = make_newton(samples=1_000, G=G)

    X = data.drop(columns="f")
    y = data.f
    threshold = np.mean(y)
    y_c = [1 if a_ > threshold else 0 for a_ in y]

    X_test = test.drop(columns="f")
    y_test = test.f
    y_test_c = [1 if a_ > threshold else 0 for a_ in y_test]
    nc = NewtonClassifier(threshold=np.mean(y))

    nc.fit(X, y_c)
    print(np.sum(nc.predict(X)))
