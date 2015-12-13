import random
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from SMO.helpers import Kernels
from SMO.helpers.Stats import Stats

# INPUT_DATA = '../data/easy_chips.txt'
INPUT_DATA = '../data/chips.txt'

DIMS = 2

# sigmoid = lambda x: 1 / (1 + math.exp(-x))


# sigmoid = lambda x: np.arctan(x)

DEBUG_OUTPUT = 0
C = 1.0
SIGMA = 0.025

MAX_PASSES = 50
TOL = 1e-3
EPS = 1e-5


def K(x1, x2):
    # return Kernels.gaussian(Dists.euclid(x1, x2))
    return Kernels.gaussian_kernel(x1, x2, sigma=SIGMA)


def get_result(x_row, a, b, x, y):
    return sum(a[i] * y[i] * K(x[i], x_row) for i in range(len(y))) + b


def process_one(d_train, d_test):
    np.random.shuffle(d_train)
    np.random.shuffle(d_test)
    x = d_train[:, :DIMS]
    y = d_train[:, DIMS]

    # train
    m = len(d_train)
    a = np.zeros((m,))
    b = 0.0

    passes = 0
    while passes < MAX_PASSES:
        num_changed_alphas = 0
        for i in range(m):
            ei = get_result(x[i], a, b, x, y) - 1.0 * y[i]
            pt = 0
            if (y[i] * ei < -TOL and a[i] < C) or (y[i] * ei > TOL and a[i] > 0):
                j = random.randint(0, m - 1)
                while j == i:
                    j = random.randint(0, m - 1)
                ej = get_result(x[j], a, b, x, y) - y[j]
                old_ai, old_aj = a[i], a[j]
                if y[i] != y[j]:
                    L = max(0, a[j] - a[i])
                    H = min(C, C + a[j] - a[i])
                else:
                    L = max(0, a[i] + a[j] - C)
                    H = min(C, a[i] + a[j])
                if L == H:
                    continue
                nu = 2 * K(x[i], x[j]) - K(x[i], x[i]) - K(x[j], x[j])
                if nu >= 0:
                    continue
                a[j] -= y[j] * (ei - ej) / nu
                if a[j] > H:
                    a[j] = H
                elif a[j] < L:
                    a[j] = L

                if abs(a[j] - old_aj) < EPS:
                    continue
                a[i] += y[i] * y[j] * (old_aj - a[j])

                b1 = b - ei - y[i] * (a[i] - old_ai) * K(x[i], x[i]) - y[j] * (a[j] - old_aj) * K(x[i], x[j])
                b2 = b - ej - y[i] * (a[i] - old_ai) * K(x[i], x[j]) - y[j] * (a[j] - old_aj) * K(x[j], x[j])
                if 0 < a[i] < C:
                    b = b1
                elif 0 < a[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

    # test
    x = d_test[:, :DIMS]
    y = d_test[:, DIMS]

    if DEBUG_OUTPUT:
        print('a:', a)
        print('b:', b)

    P = np.array([get_result(x_row, a, b, d_train[:, :DIMS], d_train[:, DIMS]) >= 0.0 for x_row in x])
    P_ans = P
    y_bool = y >= 0.0

    s = Stats(P_ans, y_bool)
    return s.__dict__


CROSS_K = 3
CROSS_T = 3

TRAIN_PART = 0.9


def main():
    data = np.genfromtxt(INPUT_DATA, delimiter='\t', skip_header=1)
    # data = np.genfromtxt(INPUT_DATA, delimiter=' ', skip_header=1 )
    for i in range(len(data)):
        if data[i, 2] == 0:
            data[i, 2] = -1
    train_n = int(len(data) * TRAIN_PART)

    dt = pd.DataFrame()

    for n in range(CROSS_K):
        np.random.shuffle(data)
        print("________")
        print("Cross #", n)  # , "Ker: ", ker_name)
        print('--- average from', CROSS_T, '---')
        x = defaultdict(lambda: 0.0)
        for j in range(CROSS_T):
            cur_x = process_one(data[:train_n], data[:train_n])
            for k, v in cur_x.items():
                x[k] += v
        x = {k: (v / CROSS_T) for k, v in x.items()}

        # print(x)
        interest_x = {k: v for k, v in x.items() if k in ['accuracy', 'f_measure']}
        print(interest_x)
        print(x)
        # dt.append(pd.DataFrame(x, index=['accuracy', 'f_measure']))
        # print(dt.to_clipboard())


if __name__ == '__main__':
    main()
