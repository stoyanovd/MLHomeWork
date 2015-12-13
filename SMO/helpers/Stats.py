from collections import defaultdict


class Stats:
    def __init__(self, ans, y):
        assert len(ans) == len(y)
        self.n = len(y)
        # self.__dict__ = defaultdict(lambda: 0)
        # [print(ans[i], 'vs', y[i], '(', ans[i] == y[i], ')') for i in range(self.n)]
        self.TP = ((ans == y) & ans).sum()
        self.FP = ((ans != y) & ans).sum()
        self.TN = ((ans == y) & (~ans)).sum()
        self.FN = ((ans != y) & (~ans)).sum()
        try:
            self.accuracy = (self.TP + self.TN) * 1.0 / self.n
            self.precision = self.TP / (self.TP + self.FP)
            self.recall = self.TP / (self.TP + self.FN)
            self.f_measure = 2 / (1 / self.precision + 1 / self.recall)
        except ZeroDivisionError as e:
            print("!!! Sth goes wrong.")
            print(e)
