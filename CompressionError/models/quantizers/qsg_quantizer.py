"""
    Implementation of the paper
    Dan Alistarh, Demjan Grubic, Ryota Tomioka, and Milan Vojnovic,
    'QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding', NIPS 2017

    QSGD quantizer:
    q, scale =  _qsgd_quantizer(x, s, seed=None, order=np.inf):

    Dequantizer:
    y = scale * q / s
"""

import numpy as np


class qsg_quantizer:
    def __init__(self, bucket_size, num_levels):
        self._bucket_size = bucket_size
        self._num_levels = num_levels

    def quantize(self, X, reconstructed=True):
        """
        quantize input tensor W using QSGD method. the input tensor is reshaped into vecot form and divided into buckets
        of length d. it used maximum value of the vector as the scaling parameter for quantization. The output scale is
        such that by multiplying it with quantized values, the points will be reconstructed.
        :param W: input tensor to be quantizer
        :param d: bucket size
        :param num_levels: number of levels for quantizing |W|
        :return: quantized values and the scale
        """

        if self._bucket_size is None:
            self._bucket_size = X.size

        if X.size % self._bucket_size != 0:
            raise ValueError('the number of variables must be divisible by the bucket size (d).')

        w = np.reshape(X, newshape=(-1, self._bucket_size))
        norm_w = np.linalg.norm(w, ord=np.inf, axis=1) + np.finfo(float).eps

        # 1- normalize w
        sign_w = np.sign(w)
        y = np.abs(w) / norm_w[:, np.newaxis]

        # 2- initial quantization (q0(y) = l where y is in [l/s, (l+1)/s)
        q0 = np.floor(y * self._num_levels)    # an integer number in the range 0, 1, ..., s
        # d is the normalized distance of each point to the left boundary of the quantization interval
        d = self._num_levels * y - q0

        # 3- create random binary numbers, b_i = 0 with probability (1-d) and b_i = 1 with probability d
        b = np.zeros(shape=w.shape)
        b[np.random.random(size=w.shape) < d] = 1

        q = sign_w * (q0 + b)
        scale = norm_w / self._num_levels

        if reconstructed:
            wh = q * scale[:, np.newaxis]
            Xh = np.reshape(wh, newshape=X.shape)

            return Xh
        else:
            Q = np.reshape(q, newshape=X.shape).astype(np.int)

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

        if self._bucket_size == Q.size:
            Xh = scale[0] * Q
        else:
            q = np.reshape(Q, (-1, self._bucket_size))
            w = q * scale[:, np.newaxis]

            Xh = np.reshape(w, newshape=Q.shape)

        return Xh

    def reset(self):
        return
