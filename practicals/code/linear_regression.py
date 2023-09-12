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
    Predict y_hat values with linear regression. 
    :param X: Input data matrix
    :param beta: Previously estimated coefficient vector for the linear regression 

    :return: The predicted values of the target vector from the linear regression of the input data matrix
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate y estimates
    # rows of X correspond to xi in slides, so X doesn't need to be transformed for 'row-vectors' 
    y_hat = np.dot(X, beta) 

    return y_hat

def mean_squared_error(X, y_real, beta):
    """
    Apply the predict function to perform the linear regression
    and compute mean squared error over prediction with linear regression with the real values.
    :param X: Input data matrix
    :param y_real: Real values of the target vector
    :param beta: Previously estimated coefficient vector for the linear regression 

    :return: The mean squared error of the linear regression with the given estimated coefficient vector
    """
    # predict y_hat
    y_hat = predict(X, beta)

    # calculate the mean squared error between estimates and real y values
    mse = np.sum(np.square(y_real - y_hat)) / y_real.shape[0]

    return mse