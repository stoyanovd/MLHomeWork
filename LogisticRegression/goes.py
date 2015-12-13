from collections import defaultdict
from functools import partial
import math
from scipy import spatial
import numpy as np
import pandas as pd
from LogisticRegression.helpers import Kernels
from LogisticRegression.helpers.DataKeeper import DataKeeper
from LogisticRegression.helpers.Stats import Stats

LOGISTIC_DATA = '../data/Logistic/chips.txt'


def α(step):
    return 0.2 ** (-step/10)


DIMS = 2

sigmoid = lambda x: 1 / (1 + math.exp(-x))


# sigmoid = lambda x: np.arctan(x)

def read_data():
    data = np.genfromtxt(LOGISTIC_DATA, delimiter=',')


def custom_dist(θ, xj):
    # return sum(θ[i] * xj[i] for i in range(DIMS))
    return sum((θ[i] * xj[i]) ** 2 for i in range(DIMS))


STEPS_COUNT = 500


def process_one(d_train, d_test):
    np.random.shuffle(d_train)
    np.random.shuffle(d_test)
    x = d_train[:, :DIMS]
    y = d_train[:, DIMS]

    def step_grad(θ, step):
        return α(step) * sum((y[j] - sigmoid(custom_dist(θ, x[j]))) * x[j] for j in range(len(d_train)))

    θ = np.random.random_sample(DIMS)
    for i in range(STEPS_COUNT):
        θ += step_grad(θ, i)

    x = d_test[:, :DIMS]
    y = d_test[:, DIMS]

    P = np.array([sigmoid(custom_dist(θ, x_row)) for x_row in x])
    P_ans = P >= 0.5

    s = Stats(P_ans, y)
    return s.__dict__


CROSS_K = 3
CROSS_T = 3

TRAIN_PART = 0.9


def main():
    data = np.genfromtxt(LOGISTIC_DATA, delimiter=',')
    train_n = int(len(data) * TRAIN_PART)

    dt = pd.DataFrame()

    for n in range(CROSS_K):
        np.random.shuffle(data)
        print("________")
        print("Cross #", n)  # , "Ker: ", ker_name)
        print('--- average from', CROSS_T, '---')
        x = process_one(data[:train_n], data[:train_n])
        for j in range(CROSS_T):
            cur_x = process_one(data[:train_n], data[:train_n])
            for k, v in cur_x.items():
                x[k] += v
        r = dict()
        x = {k: (v / CROSS_T) for k, v in x.items()}

        # print(x)
        interest_x = {k: v for k, v in x.items() if k in ['accuracy', 'f_measure']}
        print(interest_x)
        # dt.append(pd.DataFrame(x, index=['accuracy', 'f_measure']))
        # print(dt.to_clipboard())


if __name__ == '__main__':
    main()
