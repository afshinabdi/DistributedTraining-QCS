"""
    Implementation of dithered quantizer
"""

import tensorflow as tf


# =============================================================================
# dithered quantization
def quantize(W, num_levels, bucket_size, seed):
    if num_levels == 0:
        # for sign-based quantization
        sign_quantizer = True
        num_levels = 0.5
    else:
        sign_quantizer = False

    # reshape to the bucket
    w = tf.reshape(W, shape=[-1, bucket_size])
    w_shape = tf.shape(w)

    # generate random signals: dither signals and Rademacher variables
    u = tf.random.uniform(shape=w_shape, minval=-0.5, maxval=0.5, dtype=tf.float32, seed=seed)

    # normalize w to become in [-num_levels, num_levels]
    max_w = tf.reduce_max(tf.abs(w), axis=1, keepdims=True) + 1e-12
    scale = max_w / num_levels
    y = w / scale

    # generate dither, add it to y and then quantize
    if sign_quantizer:
        q = tf.cast((y + u) > 0, tf.float32)
    else:
        q = tf.round(y + u)

    wh = (q - u) * scale

    # dequantize operations
    Wh = tf.reshape(wh, shape=tf.shape(W))

    return q, scale, Wh


def dequantize(q, scale, seed):
    """
    dequantize the received quantized values, usign the bucket size d and scales
    :param Q: quantized values
    :param scale: scale to multiply to the quantized values to reconstruct the original data
    :param d: bucket size
    :return: ndarray of the same shape as Q, dequantized values
    """

    w_shape = tf.shape(q)

    # generate random dither signals
    u = tf.random.uniform(shape=w_shape, minval=-0.5, maxval=0.5, dtype=tf.float32, seed=seed)

    w = (q - u) * scale

    return w