"""
    Implementation of the paper 'Sparsified SGD with Memory'
    This is mainly based on the code available at 'https://github.com/epfml/sparsifiedSGD'
    especially the file memory.py
"""

import numpy as np
import tensorflow as tf

def topk_sgd(W, k):
    """
        Top-K SGD sparsification with memory
    """

    W_shape = W.get_shape().as_list()
    W_size = np.prod(W_shape)
    k = min(np.prod(W_shape), k)

    w = tf.reshape(W, shape=(-1,))
    residue = tf.Variable(tf.zeros(shape=(W_size,)), dtype=tf.float32, trainable=False)

    x = w + residue
    _, indices = tf.math.top_k(tf.abs(x), k, sorted=False)

    new_residue = tf.tensor_scatter_update(x, tf.expand_dims(indices, 1), tf.zeros(k, tf.float32))
    xh = x - new_residue
    Wh = tf.reshape(xh, W_shape)
    with tf.control_dependencies([Wh, new_residue]):
        update_residue = residue.assign(new_residue)

    return Wh, update_residue, residue

