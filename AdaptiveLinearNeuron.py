# 梯度相反的方向是学习速率最快的方向

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AdaLinNeuGD(object):
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        '''
        Fit training data
        :param X: shape = [n_samples, n_features]
        :param y: shape = n_sample
        :return: self: Object
        '''
        self.w_ = np.zeros(1 + X.shape[1])   # Initial weights
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)   # Update weights
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0    # Check whether convergence
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, 0)


def plot_adjust_parameters(X, y):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ada1 = AdaLinNeuGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Log(Sum-squared-error')
    ax[0].set_title('Adaline-Learning rate 0.01')

    ada2 = AdaLinNeuGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_)+1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline-Learning rate 0.0001')
    plt.show()

    return