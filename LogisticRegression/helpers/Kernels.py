import math
import numpy as np
import numpy.linalg as LA

squared = lambda x: 1.0 / 2 / x / x


def gaussian(x, sigma):
    coef = 1 / math.sqrt(2 * math.pi) / sigma
    deg = -x * x / 2 / sigma / sigma
    return coef * math.exp(deg)


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-LA.norm(x - y) ** 2 / (2 * (sigma ** 2)))
