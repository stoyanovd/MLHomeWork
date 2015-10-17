import math

squared = lambda x: 1.0 / 2 / x / x


def gaussian(x, sigma):
    coef = 1 / math.sqrt(2 * math.pi) / sigma
    deg = -x * x / 2 / sigma / sigma
    return coef * math.exp(deg)
