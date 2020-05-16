import os
import time
import numpy as np
import scipy.io as sio
import scipy.stats as st
import tensorflow as tf
from models.Alexnet import AlexnetModel
from datasets.tfr.imagenet_tfr import ImagenetDataSet
from datasets.hadamard import load_hadamard_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

training_algorithm = 'GD'
nn_settings = {
    'initial_w': None,    # initial weights
    'initial_b': None,    # initial bias
    'training_alg': training_algorithm,    # training algorithm
    'learning_rate': 0.001,    # learning rate
    'decay_rate': 0.98,    # decay rate
    'decay_step': 500,    # decay step
    'compute_gradients': True,    # compute gradients for use in distribtued training
}

output_folder = 'QuantizedCS/Quantizer/'
num_evals = 10
batch_size = 200
layer_index = 6

db_params = {
    'data_dir': 'DataBase/Imagenet/ILSVRC2012/tfr',
    'image_size': (227, 227),
    'BGR': True,
    'one_hot': True,
    'resize_range': (256, 384),
    'num_train_samples': 1281167,
    'num_train_files': 1024,
    'train_filenames': 'train/train-{0:05d}-of-{1:05d}',
    'num_validation_samples': 50000,
    'num_validation_files': 128,
    'validation_filenames': 'validation/validation-{0:05d}-of-{1:05d}',
    'augment_training': True,
    'shuffle_buffer': 0.0001,
    'num_classes': 1000,
}

db = ImagenetDataSet(db_settings=db_params)
graph = tf.Graph()
with graph.as_default():
    imgnet_data, initializer_op = db.get_data(['train', 'validation'], batch_size=batch_size)

db_images = imgnet_data[0]
db_labels = imgnet_data[1]
db_sess = tf.Session(graph=graph)
db_sess.run(initializer_op['train'])


def train_base_model(w0=None, b0=None):
    # training is done using batch-size=256
    nn_settings['initial_w'] = w0
    nn_settings['initial_b'] = b0

    nn = AlexnetModel()
    nn.create_network(nn_settings)

    for n in range(10):
        x, y = db_sess.run([db_images, db_labels])
        nn.train(x, y)

    w0, b0 = nn.get_weights()

    return w0, b0


def evaluate_qsg(w0, b0, num_levels, bucket_size):
    # initial parameters
    nn_settings['initial_w'] = w0
    nn_settings['initial_b'] = b0

    # quantizer
    nn_settings['quantizer'] = 'qsg'
    nn_settings['bucket_sizes'] = bucket_size
    nn_settings['num_levels'] = num_levels
    nn_settings['H'] = None

    nn = AlexnetModel()
    nn.create_network(nn_settings)

    err = np.zeros(num_evals)

    for n in range(num_evals):
        x, y = db_sess.run([db_images, db_labels])
        gw, _ = nn.get_gradients(x, y)
        gwh, *_ = nn.dequantized_gradients(x, y)
        err[n] = np.linalg.norm(gwh[layer_index] - gw[layer_index]) / (np.linalg.norm(gw[layer_index]) + 1e-12)

    return err


def evaluate_dqsg(w0, b0, num_levels, bucket_size):
    # initial parameters
    nn_settings['initial_w'] = w0
    nn_settings['initial_b'] = b0

    # quantizer
    nn_settings['quantizer'] = 'dqsg'
    nn_settings['bucket_sizes'] = bucket_size
    nn_settings['num_levels'] = num_levels

    nn = AlexnetModel()
    nn.create_network(nn_settings)

    err = np.zeros(num_evals)

    for n in range(num_evals):
        x, y = db_sess.run([db_images, db_labels])
        seeds = np.random.randint(1000, 10000000, size=2 * nn.number_layers).tolist()
        gw, _ = nn.get_gradients(x, y)
        gwh, *_ = nn.dequantized_gradients(x, y, seeds)
        err[n] = np.linalg.norm(gwh[layer_index] - gw[layer_index]) / (np.linalg.norm(gw[layer_index]) + 1e-12)

    return err


def evaluate_dqtsg(w0, b0, num_levels, bucket_size, H):
    # initial parameters
    nn_settings['initial_w'] = w0
    nn_settings['initial_b'] = b0

    # quantizer
    nn_settings['quantizer'] = 'dqtsg'
    nn_settings['bucket_sizes'] = bucket_size
    nn_settings['num_levels'] = num_levels
    nn_settings['H'] = H

    nn = AlexnetModel()
    nn.create_network(nn_settings)

    err = np.zeros(num_evals)

    for n in range(num_evals):
        x, y = db_sess.run([db_images, db_labels])
        seeds = np.random.randint(1000, 10000000, size=2 * nn.number_layers).tolist()
        gw, _ = nn.get_gradients(x, y)
        gwh, *_ = nn.dequantized_gradients(x, y, seeds)
        err[n] = np.linalg.norm(gwh[layer_index] - gw[layer_index]) / (np.linalg.norm(gw[layer_index]) + 1e-12)

    return err


def test():
    bucket_size = [[288, 96], [256, 256], [384, 384], [384, 384], [256, 256], [256, 256], [256, 256], [256, 200]]

    # load hadamard matrices
    H = [[0, 0] for _ in range(len(bucket_size))]
    for layer, d in enumerate(bucket_size):
        H[layer][0] = load_hadamard_matrix(d[0])
        H[layer][1] = load_hadamard_matrix(d[1])

    for exp in range(5):
        w0, b0 = None, None
        for rep in range(5):
            # random orthonormal matrices
            T = [[0, 0] for _ in range(len(bucket_size))]
            for layer, d in enumerate(bucket_size):
                T[layer][0] = st.ortho_group.rvs(dim=d[0])
                T[layer][1] = st.ortho_group.rvs(dim=d[1])

            w0, b0 = train_base_model(w0, b0)
            err_qsg = evaluate_qsg(w0, b0, num_levels=1, bucket_size=bucket_size)
            err_dqsg = evaluate_dqsg(w0, b0, num_levels=1, bucket_size=bucket_size)
            err_dqtsgH = evaluate_dqtsg(w0, b0, num_levels=1, bucket_size=bucket_size, H=H)
            err_dqtsgT = evaluate_dqtsg(w0, b0, num_levels=1, bucket_size=bucket_size, H=T)

            print(
                'QSG: {:.3f}, DQSG: {:.3f}, DQTSG(H): {:.3f}, DQTSG(T): {:.3f}'.format(
                    np.mean(err_qsg), np.mean(err_dqsg), np.mean(err_dqtsgH), np.mean(err_dqtsgT)
                )
            )

            fname = os.path.join(output_folder, 'Alexnet/qe_%d_%d.mat' % (exp, rep))
            sio.savemat(fname, mdict={'qsg': err_qsg, 'dqsg': err_dqsg, 'dqtsgH': err_dqtsgH, 'dqtsgT': err_dqtsgT})


if __name__ == '__main__':
    test()
