import os
import random
import numpy as np
import gzip
import pickle
import math
from PIL import Image
from matplotlib import pyplot as plt


def sigmoid_logistic(x):
    return 1 / (1 + pow(math.e, -x))


def d_sigmoid_logistic(x):
    return sigmoid_logistic(x) * (1 - sigmoid_logistic(x))


N = 28 * 28  # input layer
H = 500  # hidden layer
M = 10  # output layer


def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


class DataSet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        shuffle_in_unison_scary(self.x, self.y)


class NeuralKeeper:
    f = sigmoid_logistic
    df = d_sigmoid_logistic
    loss_inner = lambda x: x * x

    f = staticmethod(f)
    df = staticmethod(df)
    loss_inner = staticmethod(loss_inner)

    nu = staticmethod(lambda x: 0.1 / (x / 100))
    L = 0.4

    def __init__(self):
        self.w1 = np.random.sample((N, H))
        self.w2 = np.random.sample((H, M))


def one_step(nk, ds, old_Q, global_iter):
    i = random.randint(0, len(ds.x) - 1)
    # forward
    ui = np.zeros((H,))
    ai = np.zeros((M,))
    eim = np.zeros((M,))
    eih = np.zeros((H,))
    ui[0] = -1
    for h in range(1, H):
        ui[h] = nk.f(sum(nk.w1[j, h] * ds.x[i, j] for j in range(N)))
    for m in range(M):
        ai[m] = nk.f(sum(nk.w2[h, m] * ui[h] for h in range(H)))
    eim = ai - (np.array(list(range(M))) == ds.y[i].astype(np.float64))
    Li = sum(map(nk.loss_inner, eim))
    print(Li)

    # backwards
    for h in range(1, H):
        eih[h] = sum(eim[m] * nk.df(nk.w2[h, m]) for m in range(M))

    # LUK!!! Remember about (-1) input
    # gradient step
    for h in range(H):
        for m in range(M):
            nk.w2[h, m] -= nk.nu(global_iter) * eim[m] * nk.df(ui[h])
    for h in range(1, H):
        for j in range(N):
            nk.w1[j, h] -= nk.nu(global_iter) * eih[h] * nk.df(ds.x[i, j])

    Q = (1 - nk.L) * old_Q + nk.L * Li
    return Q


eps = 1e-3


def one_test(nk, x_img):
    ui = np.zeros((H,))
    ai = np.zeros((M,))
    for h in range(H):
        ui[h] = nk.f(sum(nk.w1[j, h] * x_img[j] for j in range(N)))
    for m in range(M):
        ai[m] = nk.f(sum(nk.w2[h, m] * ui[h] for h in range(H)))
    return ai


TEST_ITERATION = 50
TEST_COUNT = 100


def write_img(x_img, true_y=None, ans_p=None):
    r = x_img
    r += 1.0
    r *= 255 / 2.0
    r.reshape((28, 28))
    img = Image.fromarray(x_img).convert('RGB')
    img.save('cur_img.png')


def many_tests(nk, test_set, global_iter):
    true = 0
    true_2 = 0
    for t in range(TEST_COUNT):
        i = random.randint(0, len(test_set.x) - 1)
        if t == 0:
            write_img(test_set.x[i].reshape((28, 28)))
        p = one_test(nk, test_set.x[i])
        p = sorted([(x, j) for (j, x) in enumerate(p)], reverse=True)
        if test_set.y[i] == p[0][1] or test_set.y[i] == p[1][1]:
            true_2 += 1
        if test_set.y[i] == p[0][1]:
            true += 1
    print('one and two nearest')
    print('it:{0:5d} | {1:5d} is right from {2:5d} | accuracy = {3:5.2f}'.format(global_iter, true, TEST_COUNT,
                                                                                 true * 1.0 / TEST_COUNT))
    print('it:{0:5d} | {1:5d} is right from {2:5d} | accuracy = {3:5.2f}'.format(global_iter, true_2, TEST_COUNT,
                                                                                 true_2 * 1.0 / TEST_COUNT))


def go_learn_and_test(train_set, test_set):
    nk = NeuralKeeper()
    Q = 0
    i = 1
    while True:  # not Q or Q > eps:
        Q = one_step(nk, train_set, Q, i)
        if i % TEST_ITERATION == 0:
            many_tests(nk, test_set, i)
        i += 1


# train-labels-idx1-ubyte.gz

TRAIN_N = 60 * 1000
TEST_N = 10 * 1000
TRAIN_PCK = 'train.pck'
TEST_PCK = 'test.pck'


def read_data(is_train):
    DATASET_N = TRAIN_N if is_train else TEST_N
    r = np.zeros((DATASET_N,))
    files_prefix = 'train' if is_train else 't10k'
    with gzip.open(os.path.join('..', 'data', 'neuron', 'official', files_prefix + '-labels-idx1-ubyte.gz'),
                   'rb') as f:
        f.read(2 * 4)
        f = f.read(DATASET_N)
        for i in range(DATASET_N):
            r[i] = f[i]
    print(np.unique(r))
    r = np.array(r)

    a = np.zeros((DATASET_N + 1, 28 * 28))
    a[0, :, :] = 1.0
    with gzip.open(os.path.join('..', 'data', 'neuron', 'official', files_prefix + '-images-idx3-ubyte.gz'),
                   'rb') as f:
        f.read(4 * 4)
        f = f.read(DATASET_N * 28 * 28)
        t = 0
        for i in range(1, DATASET_N + 1):
            for j in range(28):
                for h in range(28):
                    a[i, j * 28 + h] = -1 if f[t] == 0 else 1
                    t += 1

    pickle.dump((r, a), open(TRAIN_PCK if is_train else TEST_PCK, 'wb'))
    # print(type(train_set))
    # print(train_set.shape)
    # print(train_set[:5])


def restore_data(is_train):
    return pickle.load(open(TRAIN_PCK if is_train else TEST_PCK, 'rb'))


def go_with_ready_data():
    r, a = restore_data(is_train=True)
    train_set = DataSet(a, r)
    r, a = restore_data(is_train=False)
    test_set = DataSet(a, r)
    go_learn_and_test(train_set, test_set)


def main():
    # read_data(is_train=True)
    # read_data(is_train=False)
    go_with_ready_data()


if __name__ == '__main__':
    main()
