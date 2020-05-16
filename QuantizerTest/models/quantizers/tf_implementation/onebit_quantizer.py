"""
    '1-Bit Stochastic Gradient Descent and its Application to Data-Parallel Distributed Training of Speech DNNs',
    Seide, et. al. 2014
"""

import numpy as np
import tensorflow as tf


def quantize(W):
    """
    Quantizing the given matrix with only 1 bit. The threshold is fixed to zero and the reconstruction values are computed
    to minimize the MSE.
    Parameters:
        W : input data (vector or ndarray) to quantize
    """

    W_shape = W.get_shape().as_list()
    W_size = np.prod(W_shape) + 2e-10

    # variable to store the residual signal
    residue = tf.Variable(tf.zeros(shape=W_shape), dtype=tf.float32, trainable=False)

    W = W + residue
    Qp = tf.cast(W > 0, tf.float32)
    Qn = 1 - Qp
    num_p = tf.reduce_sum(Qp, axis=0, keepdims=True) + 1e-10

    W_positive = tf.multiply(W, Qp)
    W_negative = tf.multiply(W, Qn)
    centers_positive = tf.reduce_sum(W_positive, axis=0, keepdims=True) / num_p
    centers_negative = tf.reduce_sum(W_negative, axis=0, keepdims=True) / (W_size - num_p)

    # compute the quantization error
    Wh = centers_positive * Qp + centers_negative * Qn
    new_residue = W - Wh
    with tf.control_dependencies([Wh, new_residue]):
        update_residue = residue.assign(new_residue)

    return Qp, centers_positive, centers_negative, Wh, update_residue, residue


def dequantize(Q, cp, cn):
    """
    Dequanitze from the given 1-bit quantization and the reconstruction values.
    Parameters:
        Q : input quantized values (+/-  1)
        cp: center of the quantization bins for positive values
        cn: center of the quantization bins for negative values
    """

    Qn = 1 - Q
    Wh = cp * Q + cn * Qn

    return Wh
