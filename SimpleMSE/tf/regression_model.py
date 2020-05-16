'''
    Defines a simple linear regression model to analyze and compare convergence rates
'''

import numpy as np
import scipy.linalg as sla
import scipy.stats as st
import tensorflow.compat.v1 as tf    # pylint: disable=import-error
import quantizers.onebit_quantizer as obq
import quantizers.qsg_quantizer as qsg
import quantizers.cs_quantizer as csq
import quantizers.topK_sgd as topkq
import quantizers.dithered_transform_quantizer as dtq


def create_transformation(M, min_eig=1, max_eig=4):
    """
        create transformation to generate correlated Gaussain random vector
    """
    A = st.ortho_group.rvs(dim=M)
    S = np.random.random(size=M)
    S = min_eig + (max_eig - min_eig) * (S - np.min(S)) / (np.max(S) - np.min(S))
    S = np.diag(np.sqrt(S))
    T = np.matmul(S, A)    # transformation to generate correlated Gaussian random vector
    R = np.matmul(T.T, T)    # correltation matrix of input data

    return T, R


class RegressionModel:
    def __init__(self):
        self._Wshape = None
        self._batch_size = None
        self._learning_rate = None

        self._W, self._loss = None, None    # parameter and loss function
        self._gW, self._apply_gradients = None, None    # gradient of the parameter and update rule
        self._gWh, self._updateR, self._resetR = None, None, None    # quantization and residue update

        self._reset_op = None    # reset the parameters of the model to re-run

        self._sess = None

    def create(self, T, Wopt, quantizer='', **kwargs):
        self._Wshape = Wopt.shape

        # create the model
        self._create_regressor(T, Wopt)

        # define the training operations
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)
        self._trainOp = optimizer.minimize(self._loss)

        # add gradient quantizer
        self._add_gradient(quantizer, **kwargs)

        # applying input gradients to the optimizer
        self._input_gW = (tf.placeholder(dtype=tf.float32, shape=self._Wshape), )
        gv = [(self._input_gW[0], self._W)]
        self._apply_gradients = optimizer.apply_gradients(gv)

        self._reset_op = [tf.assign(self._W, np.zeros(self._Wshape)), self._resetR]

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def _create_regressor(self, T, Wopt):
        M, N = Wopt.shape

        self._batch_size = tf.placeholder(dtype=tf.int32)
        self._learning_rate = tf.placeholder(dtype=tf.float32)

        # define the linear model to fit data
        _Wopt = tf.constant(Wopt, dtype=tf.float32)
        _T = tf.constant(T, dtype=tf.float32)

        x = tf.random.normal(shape=(self._batch_size, M))
        x = tf.matmul(x, _T)
        yopt = tf.matmul(x, _Wopt)

        self._W = tf.Variable(np.zeros((M, N)), dtype=tf.float32)
        y = tf.matmul(x, self._W)
        self._loss = tf.nn.l2_loss(y - yopt) / tf.cast(self._batch_size, dtype=tf.float32)

    def _add_gradient(self, quantizer='', **kwargs):
        self._gW = tf.gradients(self._loss, self._W)
        self._updateR = tf.no_op()
        self._resetR = tf.no_op()

        feedback = kwargs.get('feedback', False)

        if quantizer == '':
            self._gWh = self._gW

        elif quantizer == 'one-bit':
            _, _, _, self._gWh, self._updateR, residue = obq.quantize(self._gW[0])
            self._resetR = tf.assign(residue, np.zeros(self._Wshape))

        elif quantizer == 'qsg':
            num_levels = kwargs.get('num_levels', 1)
            bucket_size = kwargs.get('bucket_size')
            _, _, self._gWh = qsg.quantize(self._gW[0], num_levels, bucket_size)

        elif quantizer == 'qcs':
            num_levels = kwargs.get('num_levels', 1)
            H = kwargs.get('H')
            seed = kwargs.get('seed', 73516)
            beta = kwargs.get('beta', 0)
            _, _, self._gWh, self._updateR, residue = csq.quantize(self._gW[0], H, num_levels, seed, feedback, beta)

        elif quantizer == 'topk':
            feedback = True
            K = kwargs.get('K')
            self._gWh, self._updateR, residue = topkq.topk_sgd(self._gW[0], K)

        elif quantizer == 'dtq':
            num_levels = kwargs.get('num_levels', 1)
            H = kwargs.get('H')
            seed = kwargs.get('seed', 73516)
            _, _, self._gWh = dtq.quantize(self._gW[0], H, num_levels, seed)

        if feedback:
            self._resetR = tf.assign(residue, tf.zeros_like(residue))

    # __________________________________________________________________________
    # reset the model
    def reset(self):
        self._sess.run(self._reset_op)

    def train(self, batch_size, learning_rate):
        self._sess.run(self._trainOp, feed_dict={self._batch_size: batch_size, self._learning_rate: learning_rate})

    def compute_gradients(self, batch_size):
        return self._sess.run(self._gW, feed_dict={self._batch_size: batch_size})

    def compute_quantized_gradients(self, batch_size):
        Wh, _ = self._sess.run([self._gWh, self._updateR], feed_dict={self._batch_size: batch_size})
        return Wh

    def apply_gradients(self, gW, learning_rate):
        self._sess.run(self._apply_gradients, feed_dict={self._input_gW: gW, self._learning_rate: learning_rate})

    def loss(self, batch_size):
        return self._sess.run(self._loss, feed_dict={self._batch_size: batch_size})

    @property
    def W(self):
        return self._sess.run(self._W)
