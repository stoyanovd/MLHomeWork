import numpy as np
from LinearRegression.helpers import Drawer


def process_all():
    data = np.genfromtxt('../data/prices.txt', delimiter=',')
    y = data[:, 2]

    A = data.copy()
    A = (A - A.min(0)) / (A.max(0) - A.min(0))
    A[:, [0, 1, 2]] = A[:, [2, 0, 1]]
    A[:, 0] = 1

    w = np.dot(np.linalg.inv(A.transpose().dot(A)), (A.transpose().dot(y)))
    y1 = A.dot(w)
    r = y - y1
    print(np.abs(r).mean())

    SSE = r.transpose().dot(r)
    r = r / y
    print("SSE:", SSE ) #/ (y.sum() ** 2))
    print(np.abs(r).mean())

    Drawer.draw(A[:, 1], A[:, 2], w, y)


if __name__ == '__main__':
    process_all()
