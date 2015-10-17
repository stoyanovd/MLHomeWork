import math


def euclid(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]
    return x * x + y * y


def manh(a, b):
    x = math.fabs(a[0] - b[0])
    y = math.fabs(a[1] - b[1])
    return x + y
