import os
import time
import numpy as np
import scipy.io as sio
import regression_model as rm
from hadamard import load_hadamard_matrix
import quantizers.onebit_quantizer as obq
import quantizers.qsg_quantizer as qsg
import quantizers.cs_quantizer as csq
import quantizers.topK_sgd as topkq

output_folder = 'QuantizedCS/SimpleMSE/new'
np.set_printoptions(precision=3, linewidth=80)
batch_size = 32
num_lr = 100
repeat_num = 10
num_iterations = 1000
learning_rates = np.linspace(0, 0.25, num_lr + 1)[1:]


def evaluate_baseline(T, Wopt, file_name):
    model = rm.RegressionModel(T, Wopt)

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
                g = model.gradient(batch_size)
                model.update(g, learning_rate=lr)

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
    model = rm.RegressionModel(T, Wopt)
    quantizer = obq.onebit_quantizer()

    loss = np.zeros((num_lr, repeat_num, num_iterations))
    loss2 = np.zeros((num_lr, repeat_num, num_iterations))
    for n, lr in enumerate(learning_rates):
        start = time.time()
        print('\nLearning rate = ', lr, flush=True)
        for rp in range(repeat_num):
            # create model
            model.reset()
            quantizer.reset()

            info_str = ' '
            for cnt in range(num_iterations):
                # apply one-bit quantization method to the gradients
                g = model.gradient(batch_size)
                gh = quantizer.quantize(g)
                model.update(gh, learning_rate=lr)

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


def evaluate_qsgd(T, Wopt, file_name, bucket_size, num_levels):
    # create model
    model = rm.RegressionModel(T, Wopt)
    quantizer = qsg.qsg_quantizer(bucket_size, num_levels)

    loss = np.zeros((num_lr, repeat_num, num_iterations))
    loss2 = np.zeros((num_lr, repeat_num, num_iterations))
    for n, lr in enumerate(learning_rates):
        start = time.time()
        print('\nLearning rate = ', lr, flush=True)
        for rp in range(repeat_num):
            model.reset()
            quantizer.reset()

            info_str = ' '
            for cnt in range(num_iterations):
                # apply qsgd quantization method to the gradients
                g = model.gradient(batch_size)
                gh = quantizer.quantize(g)
                model.update(gh, learning_rate=lr)

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
    model = rm.RegressionModel(T, Wopt)
    quantizer = topkq.topk_quantizer(K)

    loss = np.zeros((num_lr, repeat_num, num_iterations))
    loss2 = np.zeros((num_lr, repeat_num, num_iterations))
    for n, lr in enumerate(learning_rates):
        start = time.time()
        print('\nLearning rate = ', lr, flush=True)
        for rp in range(repeat_num):
            # create model
            model.reset()
            quantizer.reset()

            info_str = ' '
            for cnt in range(num_iterations):
                # apply qsgd quantization method to the gradients
                g = model.gradient(batch_size)
                gh = quantizer.quantize(g)
                model.update(gh, learning_rate=lr)

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


def evaluate_qcssgd(T, Wopt, file_name, H, num_levels, feedback, beta):
    model = rm.RegressionModel(T, Wopt)
    quantizer = csq.cs_quantizer(H, num_levels, feedback, beta)

    loss = np.zeros((num_lr, repeat_num, num_iterations))
    loss2 = np.zeros((num_lr, repeat_num, num_iterations))
    for n, lr in enumerate(learning_rates):
        start = time.time()
        print('\nLearning rate = ', lr, flush=True)
        for rp in range(repeat_num):
            model.reset()
            quantizer.reset()

            info_str = ' '
            for cnt in range(num_iterations):
                # apply qsgd quantization method to the gradients
                g = model.gradient(batch_size)
                gh = quantizer.quantize(g)
                model.update(gh, learning_rate=lr)

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
    M, N = 64, 32
    min_eig = 1
    max_eig = 4

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

    print('_' * 40)
    print('Evaluating baseline...')
    fname = os.path.join(output_folder, 'baseline.mat')
    evaluate_baseline(T, Wopt, file_name=fname)

    print('_' * 40)
    print('Evaluating 1-bit...')
    fname = os.path.join(output_folder, 'one_bit.mat')
    evaluate_onebit(T, Wopt, file_name=fname)

    print('_' * 40)
    print('Evaluating QSGD...')
    fname = os.path.join(output_folder, 'qsgd.mat')
    evaluate_qsgd(T, Wopt, file_name=fname, bucket_size=512, num_levels=1)

    print('_' * 40)
    print('Evaluating Top-k SGD...')
    fname = os.path.join(output_folder, 'topk20-sgd.mat')
    evaluate_topksgd(T, Wopt, file_name=fname, K=20)

    H = load_hadamard_matrix(n=512)
    k = 400
    Hk = H[:, -k:] * np.sqrt(512) / np.sqrt(k)

    print('_' * 40)
    print('Evaluating Quantized CS SGD without feedback, all H...')
    fname = os.path.join(output_folder, 'qcssgd_nfa.mat')
    evaluate_qcssgd(T, Wopt, file_name=fname, H=H, num_levels=1, feedback=False, beta=0)

    print('_' * 40)
    print('Evaluating Quantized CS SGD with feedback, all H...')
    fname = os.path.join(output_folder, 'qcssgd_wfa.mat')
    evaluate_qcssgd(T, Wopt, file_name=fname, H=H, num_levels=1, feedback=True, beta=0.1)

    print('_' * 40)
    print('Evaluating Quantized CS SGD without feedback, partial H...')
    fname = os.path.join(output_folder, 'qcssgd_nf{}_2.mat'.format(k))
    evaluate_qcssgd(T, Wopt, file_name=fname, num_levels=2, H=Hk, feedback=False, beta=0)

    print('_' * 40)
    print('Evaluating Quantized CS SGD with feedback, partial H...')
    fname = os.path.join(output_folder, 'qcssgd_wf{}(0.1)_2.mat'.format(k))
    evaluate_qcssgd(T, Wopt, file_name=fname, H=Hk, num_levels=2, feedback=True, beta=.1)


# import seaborns as sns
# ax = sns.tsplot(time="timepoint", value="BOLD signal",
#                  unit="subject", condition="ROI",
#                  data=...)

if __name__ == '__main__':
    print('Running numpy version...')
    compare_algorithms()
