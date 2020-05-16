"""
    Implementation of dithered transformed quantizer
"""

import tensorflow as tf
import tensorflow_probability as tfp

_rademacher_seed = 63509


# =============================================================================
# dithered quantization
def quantize(W, H, num_levels, seed):
    if num_levels == 0:
        # for sign-based quantization
        sign_quantizer = True
        num_levels = 0.5
    else:
        sign_quantizer = False

    bucket_size = H.shape[0]
    H = tf.constant(H, dtype=tf.float32)

    # reshape to the bucket
    w = tf.reshape(W, shape=[-1, bucket_size])
    w_shape = tf.shape(w)

    # generate random signals: dither signals and Rademacher variables
    u = tf.random.uniform(shape=w_shape, minval=-0.5, maxval=0.5, dtype=tf.float32, seed=seed)
    d = tfp.math.random_rademacher(shape=w_shape, dtype=tf.float32, seed=seed + _rademacher_seed)

    # apply the random transform on w
    w = tf.multiply(d, w)
    w = tf.matmul(w, H)

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
    # apply transform again
    wh = tf.matmul(wh, tf.transpose(H))
    wh = tf.multiply(wh, d)

    Wh = tf.reshape(wh, shape=tf.shape(W))

    return q, scale, Wh


def dequantize(q, scale, H, seed):

    H = tf.constant(H, dtype=tf.float32)

    w_shape = tf.shape(q)

    # generate random signals: dither signals and Rademacher variables
    u = tf.random.uniform(shape=w_shape, minval=-0.5, maxval=0.5, dtype=tf.float32, seed=seed)
    d = tfp.math.random_rademacher(shape=w_shape, dtype=tf.float32, seed=seed + _rademacher_seed)

    w = (q - u) * scale

    # apply transform again
    w = tf.matmul(w, tf.transpose(H))
    w = tf.multiply(w, d)

    return w
