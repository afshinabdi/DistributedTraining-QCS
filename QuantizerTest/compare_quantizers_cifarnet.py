import os
import time
import numpy as np
import scipy.io as sio
import scipy.stats as st
import tensorflow as tf
from models.CifarNet import CifarNetModel
from datasets.cifar10_dataset import Cifar10Dataset
from datasets.hadamard import load_hadamard_matrix

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

output_folder = 'QuantizedCS/Quantizer/Cifarnet'
num_evals = 10
batch_size = 128
iter_per_eval = 100

db = Cifar10Dataset(db_settings=db_params)
graph = tf.Graph()
with graph.as_default():
    db_images, db_labels, initializer_op = db.create_dataset(['train', 'test'], batch_size, 16)

db_sess = tf.Session(graph=graph)
db_sess.run(initializer_op['train'])


def evaluate_base_model():
    # training is done using batch-size=256
    nn_settings['initial_w'] = None
    nn_settings['initial_b'] = None

    nn = CifarNetModel()
    nn.create_network(nn_settings)

    for _ in range(15):
        x, y = db_sess.run([db_images, db_labels])
        nn.train(x, y)

    w0, b0 = nn.get_weights()

    et = np.zeros(num_evals)

    for n in range(num_evals):
        x, y = db_sess.run([db_images, db_labels])
        start = time.time()
        for _ in range(iter_per_eval):
            gw, gb = nn.get_gradients(x, y)

        et[n] = (time.time() - start)

    return w0, b0, et


def evaluate_qsg(w0, b0, num_levels, bucket_size):
    # initial parameters
    nn_settings['initial_w'] = w0
    nn_settings['initial_b'] = b0

    # quantizer
    nn_settings['quantizer'] = 'qsg'
    nn_settings['bucket_sizes'] = bucket_size
    nn_settings['num_levels'] = num_levels
    nn_settings['H'] = None

    nn = CifarNetModel()
    nn.create_network(nn_settings)

    et = np.zeros(num_evals)

    for n in range(num_evals):
        x, y = db_sess.run([db_images, db_labels])
        start = time.time()
        for _ in range(iter_per_eval):
            qw, sw, qb, sb = nn.quantized_gradients(x, y)

        et[n] = (time.time() - start)

    return et


def evaluate_topksg(w0, b0, K):
    # initial parameters
    nn_settings['initial_w'] = w0
    nn_settings['initial_b'] = b0

    # quantizer
    nn_settings['quantizer'] = 'topk'
    nn_settings['K'] = K

    nn = CifarNetModel()
    nn.create_network(nn_settings)

    et = np.zeros(num_evals)

    for n in range(num_evals):
        x, y = db_sess.run([db_images, db_labels])
        start = time.time()
        for _ in range(iter_per_eval):
            qw, sw, qb, sb = nn.quantized_gradients(x, y)

        et[n] = (time.time() - start)

    return et


def evaluate_dqtsg(w0, b0, num_levels, H):
    # initial parameters
    nn_settings['initial_w'] = w0
    nn_settings['initial_b'] = b0

    # quantizer
    nn_settings['quantizer'] = 'dithered-transform'
    nn_settings['num_levels'] = num_levels
    nn_settings['H'] = H

    nn = CifarNetModel()
    nn.create_network(nn_settings)

    et = np.zeros(num_evals)

    for n in range(num_evals):
        x, y = db_sess.run([db_images, db_labels])
        start = time.time()
        for _ in range(iter_per_eval):
            qw, sw, qb, sb = nn.quantized_gradients(x, y)

        et[n] = (time.time() - start)

    return et


def evaluate_qcssg(w0, b0, num_levels, H, err_feedback, feedback_beta):
    # initial parameters
    nn_settings['initial_w'] = w0
    nn_settings['initial_b'] = b0

    # quantizer
    nn_settings['quantizer'] = 'quantized-cs'
    nn_settings['num_levels'] = num_levels
    nn_settings['H'] = H
    nn_settings['error_feedback'] = err_feedback
    nn_settings['feedback_weight'] = feedback_beta

    nn = CifarNetModel()
    nn.create_network(nn_settings)

    et = np.zeros(num_evals)

    for n in range(num_evals):
        x, y = db_sess.run([db_images, db_labels])
        start = time.time()
        for _ in range(iter_per_eval):
            qw, sw, qb, sb = nn.quantized_gradients(x, y)

        et[n] = (time.time() - start)

    return et


def test():
    num_levels = 1
    bucket_size = [[320, 64], [320, 64], [384, 384], [384, 192], [192, 10]]
    layer_shapes = [[5, 5, 3, 64], [5, 5, 64, 64], [2304, 384], [384, 192], [192, 10]]

    K = [[0, 0] for _ in range(len(layer_shapes))]
    for n in range(len(layer_shapes)):
        K[n][0] = int(0.5 + np.prod(layer_shapes[n]) * np.log2(3) / 32)
        K[n][1] = int(0.5 + layer_shapes[n][-1] * np.log2(3) / 32)

    nn_settings['layer_shapes'] = layer_shapes
    nn_settings['quantizer_seeds'] = [np.random.randint(1000, 10000000, size=2 * len(layer_shapes)).tolist()]

    # load hadamard matrices
    H = [[0, 0] for _ in range(len(bucket_size))]
    Hk = [[0, 0] for _ in range(len(bucket_size))]
    for layer, d in enumerate(bucket_size):
        H[layer][0] = load_hadamard_matrix(d[0])
        H[layer][1] = load_hadamard_matrix(d[1])

    w0, b0, et_base = evaluate_base_model()
    et_qsg = evaluate_qsg(w0, b0, num_levels=num_levels, bucket_size=bucket_size)
    et_tksg = evaluate_topksg(w0, b0, K)
    et_dqtsg = evaluate_dqtsg(w0, b0, num_levels=num_levels, H=H)
    et_qcssg0 = evaluate_qcssg(w0, b0, num_levels=num_levels, H=H, err_feedback=False, feedback_beta=0)
    et_qcssg1 = evaluate_qcssg(w0, b0, num_levels=num_levels, H=H, err_feedback=True, feedback_beta=0.1)

    print(
        'baseline: {:.3f}, QSG: {:.3f}, TopK: {:.3f}, DQTSG: {:.3f}'.format(
            np.mean(et_base), np.mean(et_qsg), np.mean(et_tksg), np.mean(et_dqtsg)
        )
    )
    print('QCSSG,w/o Feedback {:.3f}, QCSSG,w/ Feedback {:.3f}'.format(np.mean(et_qcssg0), np.mean(et_qcssg1)))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fname = os.path.join(output_folder, 'gpu_run_time_%d.mat' % batch_size)
    sio.savemat(
        fname,
        mdict={
            'base': et_base,
            'qsg': et_qsg,
            'tksg': et_tksg,
            'dqtsg': et_dqtsg,
            'qcssg_nf': et_qcssg0,
            'qcssg_wf': et_qcssg1
        }
    )


if __name__ == '__main__':
    test()
