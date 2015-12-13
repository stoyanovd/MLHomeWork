import random


class DataKeeper:
    def __init__(self, data):
        self.data = data
        self.learn_len = int(0.9 * len(self.data))
        self.test_len = len(self.data) - self.learn_len
        self.learn_part = list(map(lambda a: a[0:2], self.data[0:self.learn_len]))
        self.test_part = list(map(lambda a: a[0:2], self.data[self.learn_len:]))

    def shuffle(self):
        random.shuffle(self.data)
        self.learn_part = list(map(lambda a: [a[0], a[1]], self.data[0:self.learn_len]))
        self.test_part = list(map(lambda a: [a[0], a[1]], self.data[self.learn_len:]))

    def learn_value(self, i):
        return self.data[i][2]

    def test_value(self, i):
        return self.data[self.learn_len + i][2]

