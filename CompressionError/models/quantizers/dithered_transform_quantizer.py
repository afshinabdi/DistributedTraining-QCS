"""
    Implementation of transformed dithered quantizer one dimensional quantizer
"""

import numpy as np


class dt_quantizer:
    def __init__(self, T, num_levels):
        self._T = T
        self._num_levels = num_levels
        self._bucket_size = T.shape[0]

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

        if X.size % self._bucket_size != 0:
            raise ValueError('the number of variables must be divisible by the bucket size (d).')

        if self._num_levels == 0:
            # 1-bit (sign) dithered quantization
            return self._onebit_quantizer(X, reconstructed)

        w = np.reshape(X, newshape=(-1, self._bucket_size))
        w = np.matmul(w, self._T)

        # 1- normalize x
        scale = np.linalg.norm(w, ord=np.inf, axis=1) / self._num_levels + np.finfo(float).eps

        y = w / scale[:, np.newaxis]

        # 2- generate dither, add it to y and then quantize
        u = np.random.uniform(-0.5, 0.5, size=y.shape)
        q = np.around(y + u)  # an integer number in the range -s, ..., -1, 0, 1, ..., s

        if reconstructed:
            w = (q - u) * scale[:, np.newaxis]
            w = np.matmul(w, self._T.T)
            Xh = np.reshape(w, newshape=X.shape)
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

        if Q.size % self._bucket_size != 0:
            raise ValueError('the number of variables must be divisible by the bucket size (d).')

        q = np.reshape(Q, (-1, self._bucket_size))
        u = np.random.uniform(-0.5, 0.5, size=q.shape)
        w = (q - u) * scale[:, np.newaxis]
        w = np.matmul(w, self._T.T)
        Xh = np.reshape(w, newshape=Q.shape)

        return Xh

    def _onebit_quantizer(self, X, reconstructed):
        """
        quantize input tensor W using QSG method. the input tensor is reshaped into vector form and divided into buckets of
        length d. it used maximum value of the vector as the scaling parameter for quantization. The output scale is such that
        by multiplying it with quantized values, the points will be reconstructed.
        :param W: input tensor to be quantizer
        :param d: bucket size
        :return: quantized values and the scale
        """

        w = np.reshape(X, newshape=(-1, self._bucket_size))
        w = np.matmul(w, self._T)

        # 1- normalize x
        scale = np.linalg.norm(w, ord=np.inf, axis=1) + np.finfo(float).eps

        y = w / scale[:, np.newaxis]

        # 2- generate dither, add it to y and then quantize
        u = np.random.uniform(-1., 1., size=y.shape)
        q = np.sign(y + u)  # +/- 1
        q[q == 0] = 1

        if reconstructed:
            w = (q - u) * scale[:, np.newaxis]
            w = np.matmul(w, self._T.T)
            Xh = np.reshape(w, newshape=X.shape)
            return Xh
        
        else:
            Q = np.reshape(q, newshape=X.shape).astype(int)
            return Q, scale
