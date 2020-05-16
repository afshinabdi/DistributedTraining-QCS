"""
    Implementation of the paper 'ATOMO: Communication-efficient Learning via Atomic Sparsification'
    This is mainly based on the code available at https://github.com/hwang595/ATOMO
    Since the basic (transform domain) was not available, I implemented Alg. 1.
"""

import numpy as np
import scipy.linalg as sla


class atomo_quantizer:
    def __init__(self, rank, spectral_method=True, T=None):
        self._spectral = spectral_method
        self._rank = rank
        self._T = T

    def quantize(self, X, reconstructed=True):
        if self._spectral:
            return self._spectral_atomo(X, reconstructed)
        else:
            return self._transform_atomo(X, reconstructed)

    def _spectral_atomo(self, X, reconstructed):
        orig_shape = X.shape
        if X.ndim != 2:
            X = _resize_to_2d(X)

        u, s, vT = sla.svd(X, full_matrices=False)

        i, probs = _sample_svd(s, self._rank)
        u = u[:, i]
        s = s[i] / probs
        vT = vT[i, :]

        if reconstructed:
            xh = np.dot(np.dot(u, np.diag(s)), vT)
            Xh = np.reshape(xh, newshape=orig_shape)
            return Xh
        else:
            return u, s, vT

    def _transform_atomo(self, X, reconstructed):
        """
            Original ATOMO formulation
            It assumes that transform matrix is orthonormal.
        """

        x = np.reshape(X, -1)
        coeffs = np.matmul(self._T.T, x)
        abs_c = np.abs(coeffs)
        sort_idx = np.argsort(abs_c)[::-1]
        i, probs = _atomo_probabilities(abs_c[sort_idx], self._rank)
        i = sort_idx[i]
        coeffs = coeffs[i] / probs

        if reconstructed:
            xh = np.matmul(self._T[:, i], coeffs)
            Xh = np.reshape(xh, newshape=X.shape)
            return Xh
        else:
            return i, coeffs, probs


def _resize_to_2d(x):
    """
    x.shape > 2
    If x.shape = (a, b, *c), assumed that each one of (a, b) pairs has relevant information in c.
    """
    shape = x.shape
    if x.ndim == 1:
        n = x.shape[0]
        return x.reshape((n // 2, 2))

    if all([s == 1 for s in shape[2:]]):
        return x.reshape((shape[0], shape[1]))

    # each of (a, b) has related features
    x = x.reshape((shape[0], shape[1], -1))
    # stack those related features into a tall matrix
    x_tmp = x.reshape((shape[0] * shape[1], -1))
    tmp_shape = x_tmp.shape
    return x_tmp.reshape((int(tmp_shape[0] / 2), int(tmp_shape[1] * 2)))


def _sample_svd(s, rank=0):
    if s[0] < 1e-6:
        return [0], np.array([1.0])
    probs = s / s[0] if rank == 0 else rank * s / s.sum()
    for i, p in enumerate(probs):
        if p > 1:
            probs[i] = 1
    sampled_idx = []
    sample_probs = []
    for i, p in enumerate(probs):
        #if np.random.rand() < p:
        # random sampling from bernulli distribution
        if np.random.binomial(1, p):
            sampled_idx += [i]
            sample_probs += [p]
    rank_hat = len(sampled_idx)
    if rank_hat == 0:    # or (rank != 0 and np.abs(rank_hat - rank) >= 3):
        return _sample_svd(s, rank=rank)
    return np.array(sampled_idx, dtype=int), np.array(sample_probs)


def _atomo_probabilities(coeffs, s):
    """
        Implementation of Alg. 1 in the paper.
        It is assumed that coeffs is a 1D array of sorted absolute values of the atomic representations.
        Parameters:
            coeffs (numpy 1d array) : input sort(|C|)
            s (float) : sparsity budget
    """

    n = len(coeffs)
    scale = np.sum(coeffs) + 1e-12
    probs = np.zeros(n)
    for i in range(n):
        # scale is np.sum(coeffs[i:])
        p = coeffs[i] * s / scale
        if p <= 1:
            probs[i:] = s * coeffs[i:] / scale
            break
        else:
            probs[i] = 1
            s -= 1

        # update the scale for the next iteration
        scale = scale - coeffs[i]

    sampled_idx = []
    sample_probs = []
    for i, p in enumerate(probs):
        if np.random.binomial(1, p):
            sampled_idx += [i]
            sample_probs += [p]

    return np.array(sampled_idx, dtype=int), np.array(sample_probs)
