import os
import random
from collections import defaultdict
from  math import log
from NaiveBayes.helpers import Qualities

data_path = "../data/Bayes/pu1/part"


def get_data(pack_i):
    ans = []
    for msg in os.listdir(data_path + str(pack_i)):
        with open(os.path.join(data_path + str(pack_i), msg)) as f:
            a = f.readlines()
            subject = a[0][len("Subject:"):].split()
            subject = [int(s) for s in subject]
            body = a[2].split()
            body = [int(s) for s in body]
            ans.append(('spmsg' in msg, subject, body))
    return ans


def process_one(k):
    d = defaultdict(lambda: 0)
    a = get_data(k)
    random.shuffle(a)
    train = a[:int(0.9 * len(a))]
    spm_count = sum(int(p[0]) for p in train)
    for (label, s, b) in train:
        for sw in s:
            d[label, 0, sw] += 1 / spm_count
        for bw in b:
            d[label, 1, bw] += 1 / spm_count
    classes = {True: spm_count / len(train), False: (len(train) - spm_count) / len(train)}

    ans = defaultdict(lambda: 0)
    test = a[int(0.9 * len(a)):]
    for (label, s, b) in test:
        prediction = min([False, True],
                         key=lambda cl: -log(classes[cl]) +
                                        sum(-log(max(d[cl, 0, sw], 10 ** (-7))) for sw in s) +
                                        sum(-log(max(d[cl, 1, bw], 10 ** (-7))) for bw in b))
        ans[label == prediction, prediction] += 1

    return ans


cross_k = 10


def process_all():
    for pack_i in range(1, 11):
        a = 0.0
        f = 0.0
        for i in range(cross_k):
            d = process_one(pack_i)
            a += Qualities.accuracy(d) / cross_k
            f += Qualities.fMeasure(d) / cross_k
        print("Pack:", pack_i, "(average from ", cross_k, ") Accuracy:", a, "F1score:", f)


if __name__ == '__main__':
    process_all()
