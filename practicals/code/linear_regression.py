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

def mean_squared_error(X, y_real, beta):

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate y estimates
    # rows of X correspond to xi in slides, so X doesn't need to be transformed for 'row-vectors' 
    y_pred = np.dot(X, beta) 

    # calculate the mean squared error between estimates and real y values
    mse = np.sum(np.square(y_real - y_pred)) / y_real.shape[0]

    return mse