import os
import time
import numpy as np
import scipy.io as sio
import scipy.stats as st
import tensorflow as tf
from models.CifarNet import CifarNetModel
from datasets.cifar10_dataset import Cifar10Dataset
from datasets.hadamard import load_hadamard_matrix

# quantizers
import quantizers.cs_quantizer as np_csq
import quantizers.qsg_quantizer as np_qsg
import quantizers.dithered_transform_quantizer as np_dtq
import quantizers.onebit_quantizer as np_obq
import quantizers.topK_sgd as np_topK
import quantizers.atomo as np_atomo

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

training_algorithm = 'GD'
nn_settings = {
    'initial_w': None,    # initial weights
    'initial_b': None,    # initial bias
    'training_alg': training_algorithm,    # training algorithm
    'learning_rate': 0.2,    # learning rate
    'decay_rate': 0.98,    # decay rate
    'decay_step': 500,    # decay step
    'compute_gradients': True,    # compute gradients for use in distribtued training
}

db_params = {
    'database-dir': 'Database/CIFAR10/raw',
    'one-hot': False,
    'output-dimension': (24, 24),
    'augment-training': True,
}

output_folder = 'QuantizedCS/Quantizer/Cifarnet/mse'
num_evals = 20
batch_size = 256
layer_index = 2

db = Cifar10Dataset(db_settings=db_params)
graph = tf.Graph()
with graph.as_default():
    db_images, db_labels, initializer_op = db.create_dataset(['train', 'test'], batch_size, 16)

db_sess = tf.Session(graph=graph)
db_sess.run(initializer_op['train'])


def train_base_model(w0=None, b0=None):
    # training is done using batch-size=256
    nn_settings['initial_w'] = w0
    nn_settings['initial_b'] = b0

    nn = CifarNetModel()
    nn.create_network(nn_settings)

    for _ in range(150):
        x, y = db_sess.run([db_images, db_labels])
        nn.train(x, y)

    w0, b0 = nn.get_weights()

    return w0, b0


def evaluate_qsg(nn, bucket_size, fname):
    w0, b0 = nn.get_weights()
    input_bits = w0[layer_index].size * 32
    Q = np.arange(1, 50)
    err = np.zeros((len(Q), num_evals))
    compression_gain = np.zeros(len(Q))

    for nq, q in enumerate(Q):
        quantizer = np_qsg.qsg_quantizer(bucket_size, q)
        # compute compression gain
        x, y = db_sess.run([db_images, db_labels])
        gw, _ = nn.get_gradients(x, y)
        gw = gw[layer_index]
        v, s = quantizer.quantize(gw, reconstructed=False)
        compression_gain[nq] = input_bits / (32 * s.size + np.log2(2 * q + 1) * v.size)

        # compute error
        for n in range(num_evals):
            x, y = db_sess.run([db_images, db_labels])
            gw, _ = nn.get_gradients(x, y)
            gw = gw[layer_index]
            gwh = quantizer.quantize(gw, reconstructed=True)
            err[nq, n] = np.linalg.norm(gwh - gw) / np.linalg.norm(gw)

    sio.savemat(fname, mdict={'cg': compression_gain, 'err': err, 'Q': Q})


def evaluate_topksg(nn, fname):
    w0, b0 = nn.get_weights()
    input_bits = w0[layer_index].size * 32
    maxK = w0[layer_index].size // 5
    K = np.arange(1, maxK, 10)
    compression_gain = np.zeros(len(K))
    err = np.zeros((len(K), num_evals))

    for nk, k in enumerate(K):
        quantizer = np_topK.topk_quantizer(k)
        # compute compression gain
        x, y = db_sess.run([db_images, db_labels])
        gw, _ = nn.get_gradients(x, y)
        gw = gw[layer_index]
        ind, v = quantizer.quantize(gw, reconstructed=False)
        compression_gain[nk] = input_bits / (8 * (ind[0].size + ind[1].size) + 32 * v.size)

        # compute error
        for n in range(num_evals):
            x, y = db_sess.run([db_images, db_labels])
            gw, _ = nn.get_gradients(x, y)
            gw = gw[layer_index]
            gwh = quantizer.quantize(gw, reconstructed=True)
            err[nk, n] = np.linalg.norm(gwh - gw) / np.linalg.norm(gw)

    sio.savemat(fname, mdict={'cg': compression_gain, 'err': err, 'K': K})


def evaluate_atomo(nn, fname):
    w0, b0 = nn.get_weights()
    input_bits = w0[layer_index].size * 32
    maxR = np.prod(w0[layer_index].shape[:-1])
    R = np.arange(1, maxR, 10)
    compression_gain = np.zeros(len(K))
    err = np.zeros((len(K), num_evals))

    for nk, k in enumerate(R):
        quantizer = np_atomo.atomo_quantizer(k, True)

        # compute compression gain
        x, y = db_sess.run([db_images, db_labels])
        gw, _ = nn.get_gradients(x, y)
        gw = gw[layer_index]
        u, v, s = quantizer.quantize(gw, reconstructed=False)
        compression_gain[nk] = input_bits / (32 * (u.size + v.size + s.size))

        # compute error
        for n in range(num_evals):
            x, y = db_sess.run([db_images, db_labels])
            gw, _ = nn.get_gradients(x, y)
            gw = gw[layer_index]
            gwh = quantizer.quantize(gw, reconstructed=True)
            err[nk, n] = np.linalg.norm(gwh - gw) / np.linalg.norm(gw)

    sio.savemat(fname, mdict={'cg': compression_gain, 'err': err, 'R': R})


def evaluate_dqtsg(nn, H, fname):
    w0, b0 = nn.get_weights()
    input_bits = w0[layer_index].size * 32
    Q = np.arange(1, 50)
    err = np.zeros((len(Q), num_evals))
    compression_gain = np.zeros(len(Q))

    for nq, q in enumerate(Q):
        quantizer = np_dtq.dt_quantizer(H, q)
        # compute compression gain
        x, y = db_sess.run([db_images, db_labels])
        gw, _ = nn.get_gradients(x, y)
        gw = gw[layer_index]
        v, s = quantizer.quantize(gw, reconstructed=False)
        compression_gain[nq] = input_bits / (32 * s.size + np.log2(2 * q + 1) * v.size)

        # compute error
        for n in range(num_evals):
            x, y = db_sess.run([db_images, db_labels])
            gw, _ = nn.get_gradients(x, y)
            gw = gw[layer_index]
            gwh = quantizer.quantize(gw, reconstructed=True)
            err[nq, n] = np.linalg.norm(gwh - gw) / np.linalg.norm(gw)

    sio.savemat(fname, mdict={'cg': compression_gain, 'err': err, 'Q': Q})


def evaluate_qcssg(nn, H, fname):
    w0, b0 = nn.get_weights()
    input_bits = w0[layer_index].size * 32
    maxK = H.shape[0]
    K = np.arange(1, maxK, 10)
    Q = np.arange(1, 5)

    compression_gain = np.zeros((len(K), len(Q)))
    err = np.zeros((len(K), len(Q), num_evals))
    for nk, k in enumerate(K):
        print(k/maxK, flush=True)
        Hk = H[:, -k:] * np.sqrt(maxK) / np.sqrt(k)
        for nq, q in enumerate(Q):
            quantizer = np_csq.cs_quantizer(H, q, False, 0)
            # compute compression gain
            x, y = db_sess.run([db_images, db_labels])
            gw, _ = nn.get_gradients(x, y)
            gw = gw[layer_index]
            v, s = quantizer.quantize(gw, reconstructed=False)
            compression_gain[nk, nq] = input_bits / (32 * s.size + np.log2(2 * q + 1) * s.size)

            # compute error
            for n in range(num_evals):
                x, y = db_sess.run([db_images, db_labels])
                gw, _ = nn.get_gradients(x, y)
                gw = gw[layer_index]
                gwh = quantizer.quantize(gw, reconstructed=True)
                err[nk, nq, n] = np.linalg.norm(gwh - gw) / np.linalg.norm(gw)

    sio.savemat(fname, mdict={'cg': compression_gain, 'err': err, 'K': K, 'Q': Q})


def test():
    bucket_sizes = [[320, 64], [320, 64], [384, 384], [384, 192], [192, 10]]
    layer_shapes = [[5, 5, 3, 64], [5, 5, 64, 64], [2304, 384], [384, 192], [192, 10]]
    bucket_size = bucket_sizes[layer_index][0]
    layer_shape = layer_shapes[layer_index]

    # load hadamard matrices
    H = load_hadamard_matrix(n=bucket_size)

    # load/train initial model
    model_fname = os.path.join(output_folder, 'model.npz')
    if not os.path.exists(model_fname):
        w0, b0 = train_base_model()
        np.savez(model_fname, *w0, *b0)
    else:
        data = np.load(model_fname, encoding='latin1')
        keys = np.sort(list(data.keys()))
        num_layers = len(keys) // 2
        w0 = [data[keys[k]] for k in range(num_layers)]
        b0 = [data[keys[k]] for k in range(num_layers, 2 * num_layers)]
        data.close()

    # create the neural network
    nn_settings['initial_w'] = w0
    nn_settings['initial_b'] = b0

    nn = CifarNetModel()
    nn.create_network(nn_settings)

    db_sess.run(initializer_op['test'])
    x, y = db_sess.run([db_images, db_labels])
    print('Model accuracy: ', nn.accuracy(x, y))

    db_sess.run(initializer_op['train'])

    # evaluate QSG
    # fname = os.path.join(output_folder, 'qsg.mat')
    # evaluate_qsg(nn, bucket_size, fname)

    # evaluate dithered transformed sg
    # fname = os.path.join(output_folder, 'dqtsg.mat')
    # evaluate_dqtsg(nn, H, fname)

    # evaluate quantized compressive sampling
    fname = os.path.join(output_folder, 'qcssg.mat')
    evaluate_qcssg(nn, H, fname)

    # evaluate top-k sg
    fname = os.path.join(output_folder, 'topk.mat')
    evaluate_topksg(nn, fname)

    # evaluate spectral atomo
    fname = os.path.join(output_folder, 'sp_atomo.mat')
    evaluate_atomo(nn, fname)


if __name__ == '__main__':
    test()
