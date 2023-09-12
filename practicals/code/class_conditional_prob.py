#https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

#load data
breast_cancer = load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target[:, np.newaxis]
X2 = np.append(X, Y, axis=1)
print(X[:5])
print(Y[:5])

df_X = pd.DataFrame(X)
df_Y = pd.DataFrame(Y)

#Function for creating distributions
def fit_dist(data):
    mu=np.mean(data)
    sigma = np.std(data)
    print(mu, sigma)
    dist = norm(mu, sigma)
    return dist

#Calculating prior probabilities
X2_0 = X2[X2[:, 30] == 0]
X2_1 = X2[X2[:, 30] == 1]
p_y0 = len(X2_0) / len(X)
p_y1 = len(X2_1) / len(X)

print(X2_0.shape, X2_1.shape, p_y0, p_y1)

#PDF's for y = 0
print(fit_dist(X[:,0]))

#PDF's for y = 1

