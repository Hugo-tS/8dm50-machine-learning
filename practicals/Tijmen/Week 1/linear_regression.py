import numpy as np

def lsq(X, y):
    """
    Least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :return: Estimated coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta

def predict(X, beta):
    """
    Predicts y_hat for new X by adding the same bias column.
    X: (m, d)
    beta: (d+1, 1)
    returns y_hat: (m, 1)
    """
    ones = np.ones((len(X), 1))
    Xb = np.concatenate((ones, X), axis=1)         # (m, d+1)
    return Xb @ beta                               # (m, 1)

def mse(y_true, y_pred):
    """
    Mean squared error.
    Accepts column vectors or 1D.
    """
    yt = np.asarray(y_true).reshape(-1, 1)
    yp = np.asarray(y_pred).reshape(-1, 1)
    return np.mean((yt - yp) ** 2)

