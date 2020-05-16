'''
    Defines a simple linear regression model to analyze and compare convergence rates
'''

import numpy as np
import scipy.linalg as sla
import scipy.stats as st


def create_transformation(M, min_eig=1, max_eig=4):
    """
        create transformation to generate correlated Gaussain random vector
    """
    A = st.ortho_group.rvs(dim=M)
    S = np.random.random(size=M)
    S = min_eig + (max_eig - min_eig) * (S - np.min(S)) / (np.max(S) - np.min(S))
    S = np.diag(np.sqrt(S))
    T = np.matmul(S, A)    # transformation to generate correlated Gaussian random vector
    R = np.matmul(T.T, T)    # correltation matrix of input data

    return T, R


class RegressionModel:
    def __init__(self, T, Wopt):
        self._Wshape = Wopt.shape
        self._T = T
        self._Wopt = Wopt
        self._W = np.zeros_like(self._Wopt)

    def reset(self):
        self._W = np.zeros_like(self._Wopt)

    def _create_data_samples(self, batch_size):
        x = np.random.normal(0, 1, size=(batch_size, self._Wopt.shape[0]))
        x = np.matmul(x, self._T)
        y = np.matmul(x, self._Wopt)

        return x, y

    def loss(self, batch_size):
        x, yd = self._create_data_samples(batch_size)
        y = np.matmul(x, self._W)
        e = yd - y

        return np.sum(e**2) / batch_size / 2

    def gradient(self, batch_size):
        x, yd = self._create_data_samples(batch_size)
        y = np.matmul(x, self._W)
        g = np.matmul(x.T, y - yd) / batch_size

        return g

    def update(self, g, learning_rate):
        self._W -= g * learning_rate

    @property
    def W(self):
        return self._W
