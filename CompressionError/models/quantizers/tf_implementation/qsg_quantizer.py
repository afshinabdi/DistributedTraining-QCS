"""
    Implementation of the paper
    Dan Alistarh, Demjan Grubic, Ryota Tomioka, and Milan Vojnovic,
    'QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding', NIPS 2017

    QSGD quantizer:
    q, scale =  _qsgd_quantizer(x, s, seed=None, order=np.inf):

    Dequantizer:
    y = scale * q / s
"""

import tensorflow as tf


def quantize(W, num_levels, bucket_size):
    """
    quantize input tensor W using QSGD method. the input tensor is reshaped into vecot form and divided into buckets of
    length d. it used maximum value of the vector as the scaling parameter for quantization. The output scale is such that
    by multiplying it with quantized values, the points will be reconstructed.
    :param W: input tensor to be quantizer
    :param d: bucket size
    :param num_levels: number of levels for quantizing |W|
    :return: quantized values and the scale
    """

    w = tf.reshape(W, shape=[-1, bucket_size])
    w_shape = tf.shape(w)

    # 1- normalize w to become in [-num_levels, num_levels]
    max_w = tf.reduce_max(tf.abs(w), axis=1, keepdims=True) + 1e-12
    scale = max_w / num_levels
    y = w / scale

    # 2- generate dither, add it to y and then quantize
    u = tf.random.uniform(shape=w_shape, minval=-0.5, maxval=0.5, dtype=tf.float32)
    # an integer number in the range -num_levels, ..., num_levels
    q = tf.round(y + u)

    # dequantize
    wh = q * scale

    Wh = tf.reshape(wh, shape=tf.shape(W))
    
    return q, scale, Wh
    

def dequantize(q, scale):
    """
    dequantize the received quantized values, usign the bucket size d and scales
    :param q: quantized values
    :param scale: scale to multiply to the quantized values to reconstruct the original data
    :return: ndarray of the same shape as Q, dequantized values
    """

    w = q * scale

    return w

