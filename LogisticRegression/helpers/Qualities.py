TP = (True, True)
FP = (False, True)
FN = (False, False)
TN = (True, False)


def accuracy(d):
    return (d[TP] + d[TN]) / sum(d.values())


def precision(d):
    return d[TP] / (d[TP] + d[FP])


def recall(d):
    return d[TP] / (d[TP] + d[FP])


def fMeasure(d):
    if precision(d) + recall(d) == 0:
        return 0
    return 2 * precision(d) * recall(d) / (precision(d) + recall(d))
