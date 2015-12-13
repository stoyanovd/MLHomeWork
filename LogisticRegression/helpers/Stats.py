from collections import defaultdict


class Stats:
    def __init__(self, ans, y):
        # self.__dict__ = defaultdict(lambda: 0)
        self.TP = ((ans == y) & ans).sum()
        self.FP = ((ans != y) & ans).sum()
        self.TN = ((ans == y) & (~ans)).sum()
        self.FN = ((ans != y) & (~ans)).sum()
        try:
            self.accuracy = (self.TP + self.TN) / len(y)
            self.precision = self.TP / (self.TP + self.TN)
            self.recall = self.TP / (self.TP + self.FP)
            self.f_measure = 2 * self.precision * self.recall / (self.precision + self.recall)
        except ZeroDivisionError as e:
            print("!!! Sth goes wrong.")
            print(e)
