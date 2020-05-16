"""
    Implementation of dithered quantizer one dimensional quantizer
"""

import numpy as np


class dithered_quantizer:
    def __init__(self, bucket_size, num_levels, feedback=False):
        self._bucket_size = bucket_size
        self._num_levels = num_levels
        self._feedback = feedback
        self._residue = 0

    # dithered quantization
    def quantize(self, X, reconstructed=True):
        """
        quantize input tensor W using QSG method. the input tensor is reshaped into vector form and divided into buckets of
        length d. it used maximum value of the vector as the scaling parameter for quantization. The output scale is such that
        by multiplying it with quantized values, the points will be reconstructed.
        :param W: input tensor to be quantizer
        :param num_levels: number of levels for quantizing W, output will be in the range [-num_levels, ..., +num_levels]
        :param d: bucket size
        :return: quantized values and the scale
        """

        if self._bucket_size is None:
            self._bucket_size = X.size

        if X.size % self._bucket_size != 0:
            raise ValueError('the number of variables must be divisible by the bucket size (d).')

        if self._feedback:
            Y = X + self._residue
        else:
            Y = X

        x = np.reshape(Y, newshape=(-1, self._bucket_size))

        # 1- normalize x
        scale = np.linalg.norm(x, ord=np.inf, axis=1) / self._num_levels + np.finfo(float).eps

        y = x / scale[:, np.newaxis]

        # 2- generate dither, add it to y and then quantize
        u = np.random.uniform(-0.5, 0.5, size=y.shape)
        q = np.around(y + u)  # an integer number in the range -s, ..., -1, 0, 1, ..., s

        xh = (q - u) * scale[:, np.newaxis]
        if self._feedback:
            self._residue = Y - xh

        if reconstructed:
            Xh = np.reshape(xh, newshape=X.shape)
            return Xh

        else:
            Q = np.reshape(q, newshape=X.shape).astype(int)
            return Q, scale


    def dequantize(self, Q, scale):
        """
        dequantize the received quantized values, usign the bucket size d and scales
        :param Q: quantized values
        :param scale: scale to multiply to the quantized values to reconstruct the original data
        :param d: bucket size
        :return: ndarray of the same shape as Q, dequantized values
        """

        if self._bucket_size == Q.size:
            u = np.random.uniform(-0.5, 0.5, size=Q.shape)
            X = scale[0] * (Q - u)
        else:
            q = np.reshape(Q, (-1, self._bucket_size))
            u = np.random.uniform(-0.5, 0.5, size=q.shape)
            x = (q - u) * scale[:, np.newaxis]

            X = np.reshape(x, newshape=Q.shape)

        return X


    def reset(self):
        self._residue = 0