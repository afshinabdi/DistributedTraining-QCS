import os
import time
import numpy as np
import scipy.io as sio
import regression_model as rm
from hadamard import load_hadamard_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

output_folder = 'QuantizedCS/SimpleMSE/tf(64,50)'
np.set_printoptions(precision=3, linewidth=80)
batch_size = 32
repeat_num = 10
num_iterations = 500
num_lr = 25
learning_rates = np.linspace(0, 0.25, num_lr + 1)[1:]
# learning_rates = [0.02, 0.05, 0.06, 0.08, 0.1, 0.15]
# num_lr = len(learning_rates)


def evaluate_baseline(T, Wopt, file_name):
    model = rm.RegressionModel()
    model.create(T, Wopt, quantizer='')

    loss = np.zeros((num_lr, repeat_num, num_iterations))
    loss2 = np.zeros((num_lr, repeat_num, num_iterations))
    for n, lr in enumerate(learning_rates):
        start = time.time()
        print('\nLearning rate = ', lr, flush=True)
        for rp in range(repeat_num):
            # create model
            model.reset()

            info_str = ' '
            for cnt in range(num_iterations):
                # no quantization of the gradients
                g = model.compute_gradients(batch_size)
                model.apply_gradients(g, learning_rate=lr)

                cur_loss = model.loss(batch_size=1024)
                loss[n, rp, cnt] += cur_loss
                loss2[n, rp, cnt] += (cur_loss**2)
                if cnt % 10 == 0:
                    print(' ' * len(info_str), end='\r', flush=True)

                    if (not np.isfinite(cur_loss)) or (cur_loss > 1e10):
                        print(' Diverged.', end='\r', flush=True)
                        break

                    info_str = ' exp: {0: 2d}, iteration: {1: 4d}, loss={2:.5f}'.format(rp, cnt, cur_loss)
                    print(info_str, end='\r', flush=True)

            print('')

        elapsed = time.time() - start
        print(' elapsed time = %.3f' % elapsed, flush=True)

    sio.savemat(file_name, mdict={
        'loss': loss,
        'loss2': loss2,
        'lr': learning_rates,
    })


def evaluate_onebit(T, Wopt, file_name):
    model = rm.RegressionModel()
    model.create(T, Wopt, quantizer='one-bit')

    loss = np.zeros((num_lr, repeat_num, num_iterations))
    loss2 = np.zeros((num_lr, repeat_num, num_iterations))
    for n, lr in enumerate(learning_rates):
        start = time.time()
        print('\nLearning rate = ', lr, flush=True)
        for rp in range(repeat_num):
            # create model
            model.reset()

            info_str = ' '
            for cnt in range(num_iterations):
                # apply one-bit quantization method to the gradients
                gh = model.compute_quantized_gradients(batch_size)
                model.apply_gradients([gh], learning_rate=lr)

                cur_loss = model.loss(batch_size=1024)
                loss[n, rp, cnt] += cur_loss
                loss2[n, rp, cnt] += (cur_loss**2)
                if cnt % 10 == 0:
                    print(' ' * len(info_str), end='\r', flush=True)

                    if (not np.isfinite(cur_loss)) or (cur_loss > 1e10):
                        print(' Diverged.', end='\r', flush=True)
                        break

                    info_str = ' exp: {0: 2d}, iteration: {1: 4d}, loss={2:.5f}'.format(rp, cnt, cur_loss)
                    print(info_str, end='\r', flush=True)

            print('')

        elapsed = time.time() - start
        print(' elapsed time = %.3f' % elapsed, flush=True)

    sio.savemat(file_name, mdict={
        'loss': loss,
        'loss2': loss2,
        'lr': learning_rates,
    })


def evaluate_qsgd(T, Wopt, file_name, bucket_size):
    # create model
    model = rm.RegressionModel()
    model.create(T, Wopt, quantizer='qsg', num_levels=1, bucket_size=bucket_size)

    loss = np.zeros((num_lr, repeat_num, num_iterations))
    loss2 = np.zeros((num_lr, repeat_num, num_iterations))
    for n, lr in enumerate(learning_rates):
        start = time.time()
        print('\nLearning rate = ', lr, flush=True)
        for rp in range(repeat_num):
            model.reset()

            info_str = ' '
            for cnt in range(num_iterations):
                # apply qsgd quantization method to the gradients
                gh = model.compute_quantized_gradients(batch_size)
                model.apply_gradients([gh], learning_rate=lr)

                cur_loss = model.loss(batch_size=1024)
                loss[n, rp, cnt] += cur_loss
                loss2[n, rp, cnt] += (cur_loss**2)
                if cnt % 10 == 0:
                    print(' ' * len(info_str), end='\r', flush=True)

                    if (not np.isfinite(cur_loss)) or (cur_loss > 1e10):
                        print('Diverged.', end='\r', flush=True)
                        break

                    info_str = ' exp: {0: 2d}, iteration: {1: 4d}, loss={2:.5f}'.format(rp, cnt, cur_loss)
                    print(info_str, end='\r', flush=True)

            print('')

        elapsed = time.time() - start
        print(' elapsed time = %.3f' % elapsed, flush=True)

    sio.savemat(file_name, mdict={
        'loss': loss,
        'loss2': loss2,
        'lr': learning_rates,
    })


def evaluate_topksgd(T, Wopt, file_name, K):
    model = rm.RegressionModel()
    model.create(T, Wopt, quantizer='topk', K=K)

    loss = np.zeros((num_lr, repeat_num, num_iterations))
    loss2 = np.zeros((num_lr, repeat_num, num_iterations))
    for n, lr in enumerate(learning_rates):
        start = time.time()
        print('\nLearning rate = ', lr, flush=True)
        for rp in range(repeat_num):
            # create model
            model.reset()

            info_str = ' '
            for cnt in range(num_iterations):
                # apply qsgd quantization method to the gradients
                gh = model.compute_quantized_gradients(batch_size)
                model.apply_gradients([gh], learning_rate=lr)

                cur_loss = model.loss(batch_size=1024)
                loss[n, rp, cnt] += cur_loss
                loss2[n, rp, cnt] += (cur_loss**2)
                if cnt % 10 == 0:
                    print(' ' * len(info_str), end='\r', flush=True)

                    if (not np.isfinite(cur_loss)) or (cur_loss > 1e10):
                        print('Diverged.', end='\r', flush=True)
                        break

                    info_str = ' exp: {0: 2d}, iteration: {1: 4d}, loss={2:.5f}'.format(rp, cnt, cur_loss)
                    print(info_str, end='\r', flush=True)

            print('')

        elapsed = time.time() - start
        print(' elapsed time = %.3f' % elapsed, flush=True)

    sio.savemat(file_name, mdict={
        'loss': loss,
        'loss2': loss2,
        'lr': learning_rates,
    })


def evaluate_dtqsgd(T, Wopt, file_name, H):
    model = rm.RegressionModel()
    model.create(T, Wopt, quantizer='dtq', num_levels=1, H=H)

    loss = np.zeros((num_lr, repeat_num, num_iterations))
    loss2 = np.zeros((num_lr, repeat_num, num_iterations))
    for n, lr in enumerate(learning_rates):
        start = time.time()
        print('\nLearning rate = ', lr, flush=True)
        for rp in range(repeat_num):
            model.reset()

            info_str = ' '
            for cnt in range(num_iterations):
                # apply qsgd quantization method to the gradients
                gh = model.compute_quantized_gradients(batch_size)
                model.apply_gradients([gh], learning_rate=lr)

                cur_loss = model.loss(batch_size=1024)
                loss[n, rp, cnt] += cur_loss
                loss2[n, rp, cnt] += (cur_loss**2)
                if cnt % 10 == 0:
                    print(' ' * len(info_str), end='\r', flush=True)

                    if (not np.isfinite(cur_loss)) or (cur_loss > 1e10):
                        print('Diverged.', end='\r', flush=True)
                        break

                    info_str = ' exp: {0: 2d}, iteration: {1: 4d}, loss={2:.5f}'.format(rp, cnt, cur_loss)
                    print(info_str, end='\r', flush=True)

            print('')

        elapsed = time.time() - start
        print(' elapsed time = %.3f' % elapsed, flush=True)

    sio.savemat(file_name, mdict={
        'loss': loss,
        'loss2': loss2,
        'lr': learning_rates,
    })


def evaluate_qcssgd(T, Wopt, file_name, H, feedback, beta):
    model = rm.RegressionModel()
    model.create(T, Wopt, quantizer='qcs', num_levels=1, H=H, feedback=feedback, beta=beta)

    loss = np.zeros((num_lr, repeat_num, num_iterations))
    loss2 = np.zeros((num_lr, repeat_num, num_iterations))
    for n, lr in enumerate(learning_rates):
        start = time.time()
        print('\nLearning rate = ', lr, flush=True)
        for rp in range(repeat_num):
            model.reset()

            info_str = ' '
            for cnt in range(num_iterations):
                # apply qsgd quantization method to the gradients
                gh = model.compute_quantized_gradients(batch_size)
                model.apply_gradients([gh], learning_rate=lr)

                cur_loss = model.loss(batch_size=1024)
                loss[n, rp, cnt] += cur_loss
                loss2[n, rp, cnt] += (cur_loss**2)
                if cnt % 10 == 0:
                    print(' ' * len(info_str), end='\r', flush=True)

                    if (not np.isfinite(cur_loss)) or (cur_loss > 1e10):
                        print('Diverged.', end='\r', flush=True)
                        break

                    info_str = ' exp: {0: 2d}, iteration: {1: 4d}, loss={2:.5f}'.format(rp, cnt, cur_loss)
                    print(info_str, end='\r', flush=True)

            print('')

        elapsed = time.time() - start
        print(' elapsed time = %.3f' % elapsed, flush=True)

    sio.savemat(file_name, mdict={
        'loss': loss,
        'loss2': loss2,
        'lr': learning_rates,
    })


def compare_algorithms():
    M, N = 50, 64
    bucket_size = 320
    min_eig = 1
    max_eig = 4
    H = load_hadamard_matrix(n=bucket_size)
    k = bucket_size * 3 // 4
    Hk = H[:, -k:] * np.sqrt(bucket_size) / np.sqrt(k)
    K = int(M * N * np.log2(3) / 32)

    fname = os.path.join(output_folder, 'model.mat')
    if os.path.exists(fname):
        data = sio.loadmat(fname)
        Wopt = data['Wo']
        T = data['T']
        R = data['R']
    else:
        T, R = rm.create_transformation(M, min_eig, max_eig)
        Wopt = np.random.normal(0, 1, size=(M, N))

        sio.savemat(fname, mdict={'Wo': Wopt, 'T': T, 'R': R})

    # print('_' * 40)
    # print('Evaluating baseline...')
    # fname = os.path.join(output_folder, 'baseline.mat')
    # evaluate_baseline(T, Wopt, file_name=fname)

    # print('_' * 40)
    # print('Evaluating QSGD...')
    # fname = os.path.join(output_folder, 'qsgd.mat')
    # evaluate_qsgd(T, Wopt, file_name=fname, bucket_size=bucket_size)

    # print('_' * 40)
    # print('Evaluating Top-k SGD...')
    # fname = os.path.join(output_folder, 'topk{}-sgd.mat'.format(K))
    # evaluate_topksgd(T, Wopt, file_name=fname, K=K)

    print('_' * 40)
    print('Evaluating Quantized CS SGD without feedback, all H...')
    fname = os.path.join(output_folder, 'qcssgd_nfa.mat')
    evaluate_qcssgd(T, Wopt, file_name=fname, H=H, feedback=False, beta=0)

    # print('_' * 40)
    # print('Evaluating Quantized CS SGD with feedback, all H...')
    # fname = os.path.join(output_folder, 'qcssgd_wfa.mat')
    # evaluate_qcssgd(T, Wopt, file_name=fname, H=H, feedback=True, beta=0.5)

    # print('_' * 40)
    # print('Evaluating Quantized CS SGD without feedback, partial H...')
    # fname = os.path.join(output_folder, 'qcssgd_nf{}.mat'.format(k))
    # evaluate_qcssgd(T, Wopt, file_name=fname, H=Hk, feedback=False, beta=0)

    # print('_' * 40)
    # print('Evaluating Quantized CS SGD with feedback, partial H...')
    # fname = os.path.join(output_folder, 'qcssgd_wf{}.mat'.format(k))
    # evaluate_qcssgd(T, Wopt, file_name=fname, H=Hk, feedback=True, beta=0.2)

    # print('_' * 40)
    # print('Evaluating Dithered Quantized Transformed SGD...')
    # fname = os.path.join(output_folder, 'dtqsgd.mat')
    # evaluate_dtqsgd(T, Wopt, file_name=fname, H=H)


# import seaborns as sns
# ax = sns.tsplot(time="timepoint", value="BOLD signal",
#                  unit="subject", condition="ROI",
#                  data=...)

if __name__ == '__main__':
    compare_algorithms()
