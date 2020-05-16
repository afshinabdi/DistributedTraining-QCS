"""
    Implementation of random linear coding with error feedback
    Note that the seed of the graph must be set as well (tf.set_random_seed(.))
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

_rademacher_seed = 63509


def quantize(W, T, num_levels, seed, error_feedback=True, beta=1):
    if num_levels == 0:
        # for sign-based quantization
        sign_quantizer = True
        num_levels = 0.5
    else:
        sign_quantizer = False

    bucket_size = T.shape[0]
    W_shape = W.get_shape().as_list()
    r_shape = [np.prod(W_shape) // bucket_size, T.shape[0]]
    u_shape = [np.prod(W_shape) // bucket_size, T.shape[1]]

    if error_feedback:
        # variable to store the residual signal
        residue = tf.Variable(tf.zeros(shape=W_shape), dtype=tf.float32, trainable=False)
    else:
        residue = 0

    # transform to be applied
    T = tf.constant(T, dtype=tf.float32)

    # random dither and rademacher generator
    u = tf.random.uniform(u_shape, minval=-0.5, maxval=0.5, dtype=tf.float32, seed=seed)
    r = tfp.math.random_rademacher(shape=r_shape, dtype=tf.float32, seed=seed + _rademacher_seed)

    # A- Quantization
    if error_feedback:
        # 1- add residue to the input data
        Y = W + beta * residue
    else:
        Y = W

    # 2- reshape to the bucket
    y = tf.reshape(Y, shape=(-1, bucket_size))

    # 3- apply the transform
    y = tf.multiply(y, r)
    y = tf.matmul(y, T)

    # 4- normalize y to become in [-num_levels, num_levels]
    max_y = tf.reduce_max(tf.abs(y), axis=1, keepdims=True) + 1e-12
    scale = max_y / num_levels
    y = y / scale

    # 5- generate dither, add it to y and then quantize
    if sign_quantizer:
        q = tf.cast((y + u) > 0, tf.float32)
    else:
        q = tf.round(y + u)

    # B- Dequantization and saving residue
    # 1- dequantize
    yh = (q - u) * scale

    # 2- inverse of the transform
    yh = tf.matmul(yh, tf.transpose(T))
    yh = tf.multiply(yh, r)

    # 3- reshape
    Wh = tf.reshape(yh, shape=W_shape)

    # 4- compute and save residual signal
    if error_feedback:
        new_residue = Y - Wh
        with tf.control_dependencies([new_residue]):
            update_residue = residue.assign(new_residue)
    else:
        update_residue = tf.no_op()

    return q, scale, Wh, update_residue, residue


def dequantize(q, scale, W_shape, T, seed):
    bucket_size = T.shape[0]
    r_shape = [np.prod(W_shape) // bucket_size, T.shape[0]]
    u_shape = [np.prod(W_shape) // bucket_size, T.shape[1]]

    # transform to be applied
    T = tf.constant(T, dtype=tf.float32)

    # random dither and rademacher generator
    u = tf.random.uniform(u_shape, minval=-0.5, maxval=0.5, dtype=tf.float32, seed=seed)
    r = tfp.math.random_rademacher(shape=r_shape, dtype=tf.float32, seed=seed + _rademacher_seed)

    # 1- dequantize
    yh = (q - u) * scale

    # 2- inverse of the transform
    yh = tf.matmul(yh, tf.transpose(T))
    yh = tf.multiply(yh, r)

    # 3- reshape
    Yh = tf.reshape(yh, shape=W_shape)

    return Yh
