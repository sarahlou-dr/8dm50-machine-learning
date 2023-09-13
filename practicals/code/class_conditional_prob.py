#https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer

#load data
breast_cancer = load_breast_cancer()
X = breast_cancer.data
X_norm = (X-np.mean(X, axis = 0)) / np.std(X, axis=0)
Y = breast_cancer.target[:, np.newaxis]
X2 = np.append(X, Y, axis=1)
X2_norm = np.append(X_norm, Y, axis=1)

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
X2_0_norm = X2_norm[X2_norm[:, 30] == 0]
X2_1_norm = X2_norm[X2_norm[:, 30] == 1]
p_y0 = len(X2_0) / len(X)
p_y1 = len(X2_1) / len(X)

distX_0 = fit_dist(X2_0[:,0])
distX_0_norm = fit_dist(X2_0_norm[:,0])
distX_1 = fit_dist(X2_1[:,0])
distX_1_norm = fit_dist(X2_1_norm[:,0])


values = [value for value in np.arange(-0.2, 0.2)]

probabilities = [distX_0_norm.pdf(value) for value in values]
probabilities2 = [distX_1_norm.pdf(value) for value in values]

# plot the histogram and pdf
fig, axs = plt.subplots(5, 6)
fig.tight_layout()


k = 0
i = 0
while i < 5:
    for j in range(0, 6):           
            axs[i, j].hist(X2_0_norm[:,k], bins=10, density=True, alpha=0.7)
            axs[i, j].hist(X2_1_norm[:,k], bins=10, density=True, alpha=0.7)

            values = [value for value in np.arange(min(X_norm[:,k]), max(X_norm[:,k]))]

            probabilities = [distX_0_norm.pdf(value) for value in values]
            probabilities2 = [distX_1_norm.pdf(value) for value in values]

            axs[i, j].plot(values, probabilities2, color = "orange", label="Class 1")
            axs[i, j].plot(values, probabilities, color = "blue", label="Class 0")
            axs[i, j].set_title("Feature " + str(k+1))
            k += 1
    i += 1
    handles, labels = axs[i-1, j-1].get_legend_handles_labels()

fig.legend(handles, labels, loc='upper center')
plt.savefig("Subplots.png")
plt.show()



