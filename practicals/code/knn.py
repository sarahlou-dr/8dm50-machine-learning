import numpy as np

class knn_classifier:
    
    def __init__(self, k):
        self.k = k
        
    def fit(self, X_train, y_train):
        # normalize features to have zero mean and unit standard devation
        self.X = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
        self.y = y_train

    def predict(self, x):
        # normalize x
        x = (x - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)
        
        # first calculate all euclidean distances between x and the training data
        distances = np.sqrt(np.sum(np.power(self.X - x, 2), axis=1))

        # add indexes
        distances = [(i, distance) for i, distance in enumerate(distances)]
        
        # sort distances and select k nearest points, excluding x itself
        k_nearest = sorted(distances, key=lambda x: x[1])[1 : self.k + 1]
        
        # average the k nearest y outcomes
        y = sum(map(lambda x: self.y[x[0]], k_nearest)) / self.k
        
        # since we are doing classification, return 0 or 1 based on the mean of the k nearest outcomes
        return 0 if y <= 0.50 else 1
        
    def evaluate(self, X_test, y_test):
        # normalize test set
        x_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
        # calculate accuracy for the test set
        return sum([1 if self.predict(x) == y_test[i] else 0 for i, x in enumerate(x_test)]) / y_test.shape[0]
    
class knn_regression:
    
    def __init__(self, k):
        self.k = k
        
    def fit(self, X_train, y_train):
        # normalize features to have zero mean and unit standard devation
        self.X = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
        self.y = y_train

    def predict(self, x):
        # normalize x
        x = (x - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)
        
        # first calculate all euclidean distances between x and the training data
        distances = np.sqrt(np.sum(np.power(self.X - x, 2), axis=1))

        # add indexes
        distances = [(i, distance) for i, distance in enumerate(distances)]
        
        # sort distances and select k nearest points, excluding x itself
        k_nearest = sorted(distances, key=lambda x: x[1])[1 : self.k + 1]
        
        # average the k nearest y outcomes
        y = sum(map(lambda x: self.y[x[0]], k_nearest)) / self.k

        # since we are doing regression, return y
        return y
        
    def evaluate(self, X_test, y_test):
        # normalize test set
        x_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

        # calculate mean squared error for the test set
        return sum([np.square(self.predict(x) - y_test[i]) for i, x in enumerate(x_test)]) / X_test.shape[0]