"""
    Implementation of CS-based quantizer with error feedback
    Note that the seed of the graph must be set as well (tf.set_random_seed(.))

    Because of limitatiosn of the shuffle oeprator
        1- transpsoe of T is applied to the buckets
        2- all buckets use the same shuffled T
"""

import numpy as np


class cs_quantizer:
    def __init__(self, T, num_levels, feedback=False, beta=0):
        self._T = T
        self._num_levels = num_levels
        self._feedback = feedback
        self._beta = beta
        self._bucket_size = T.shape[0]
        self._residue = 0

        if not self._feedback:
            self._beta = 0

        if self._num_levels == 0:
            self._sign_quantizer = True
            self._num_levels = 0.5
        else:
            self._sign_quantizer = False

    def quantize(self, X, reconstructed=True):
        X_shape = X.shape
        r_shape = [X.size // self._bucket_size, self._T.shape[0]]
        u_shape = [X.size // self._bucket_size, self._T.shape[1]]

        # random dither and rademacher generator
        u = np.random.uniform(low=-0.5, high=0.5, size=u_shape)
        r = 2 * np.random.randint(low=0, high=2, size=r_shape) - 1

        # A- Quantization
        # 1- add residue to the input data
        Y = X + self._beta * self._residue

        # 2- reshape to the bucket
        y = np.reshape(Y, newshape=(-1, self._bucket_size))

        # 3- apply the transform
        y = np.matmul(y * r, self._T)

        # 4- normalize y to become in [-num_levels, num_levels]
        max_y = np.amax(np.abs(y), axis=1, keepdims=True) + 1e-12
        scale = max_y / self._num_levels
        y = y / scale

        # 5- generate dither, add it to y and then quantize
        if self._sign_quantizer:
            q = ((y + u) > 0).astype(np.float)
        else:
            q = np.round(y + u)

        # B- Dequantization and saving residue
        # 1- dequantize
        yh = (q - u) * scale

        # 2- inverse of the transform
        yh = np.matmul(yh, self._T.T)
        yh = yh * r

        # 3- reshape
        Xh = np.reshape(yh, newshape=X_shape)

        # 4- compute and save residual signal
        self._residue = Y - Xh

        if reconstructed:
            return Xh
        else:
            return q, scale

    def reset(self):
        self._residue = 0
