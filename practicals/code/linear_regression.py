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

def mse(X, y_real, beta):

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate y estimates
    y_est = np.matmul( beta.T,  X)  
    print(y_est.shape)

    y_est2 = np.dot(X, beta)

    # calculate the mean squared error between estimates and real y values
    error = (np.linalg.norm(y_real - y_est))**2 / y_real.shape[1]

    error2 = (np.linalg.norm(y_real - y_est2))**2 / y_real.shape[1]

    return error, error2