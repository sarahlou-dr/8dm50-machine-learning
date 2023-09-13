import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

def normal_distribution(X):
    """
    Compute the standard deviation, mean and their corresponding normal/Gaussian distribution of all features in X.

    :param X: input matrix with column-wise features
    :returns: a tuple with three lists with the standard deviation, mean and normal distribution for each column in X
    """
    std = np.std(X, axis=0)
    mean = np.std(X, axis=0)
    distributions = [np.random.normal(mean[i], std[i], 1000) for i in range(X.shape[1])]

    return std, mean, distributions

def pdf(std, mean, distribution):
    """
    Use the standard deviation, mean and their corresponding normal distribution to compute the probability density function.

    :param std: The standard deviation corresponding to the normal distribution
    :param mean: The mean corresponding to the normal distribution
    :param distribution: An np.random.normal distribution from which the probability density function will be determined

    :returns: a tuple of the bins sampled from the distribution and the pdf computed for it
    """
    count, bins, ignored = plt.hist(distribution, 50, density=True)
    plt.close()
    pdf = (1/(std * np.sqrt(2 * np.pi)) * np.exp( - (bins - mean)**2 / (2* std**2)))

    return bins, pdf

def plot_conditional_prob(bins_and_pdf_0, bins_and_pdf_1, distributions_0, distributions_1):
    """
    Plot the conditional probabilities of all the features in a dataset.

    :param bins_and_pdf_0: a list of tuples of the bins and pdf as computed by the function pdf for all features of a dataset for class 0
    :param bins_and_pdf_1: a list of tuples of the bins and pdf as computed by the function pdf for all features of a dataset for class 1
    :param distributions_0: a list of normal distributions computed with the function normal_distribution for all features of a dataset for class 0
    :param distributions_1: a list of normal distributions computed with the function normal_distribution for all features of a dataset for class 1

    :returns: nothing, but plots the pdf's of the different features in one figure
    """

    fig, axs = plt.subplots(5, 6, figsize=(12,10))
    fig.tight_layout()

    k = 0
    i = 0
    for i in range(0,5):
        for j in range(0, 6):           
                bins_0, pdf_0 = bins_and_pdf_0[k]
                bins_1, pdf_1 = bins_and_pdf_1[k]

                axs[i, j].hist(distributions_0[k], 50, density=True, alpha=0.7)
                axs[i, j].hist(distributions_1[k], 50, density=True, alpha=0.7)

                axs[i, j].plot(bins_0, pdf_0, color='blue', label='Class 0')
                axs[i, j].plot(bins_1, pdf_1, color='orange', label='Class 1')
                axs[i, j].set_title("Feature" + str(k+1))
                k+=1
        handles, labels = axs[i-1, j-1].get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center')