from functools import partial
import math
from scipy import spatial
from helpers import Kernels
from helpers.DataKeeper import DataKeeper
import pandas as pd
from helpers import Qualities


def read_data():
    data = []
    with open('chips.txt') as f:
        lines = f.readlines()
        for line in lines[1:]:
            numbers = line.replace(',', '.').split()
            data += [[float(numbers[0]), float(numbers[1]), int(numbers[2])]]
    return data


def predict(tree, data_keeper, p, kernel, n):
    d, ind = tree.query(p, k=n)
    a = [0.0, 0.0]
    big_dist = d[-1:]
    for i in range(n):
        part = data_keeper.learn_value(ind[i])
        a[part] += kernel(d[i] / big_dist)
    return a[1] > a[0]


TESTS_IN_PACK = 5


def run_one(data_keeper, kernel, n, ):
    ans = 0.0
    dt = []
    r_a = 0
    r_f = 0
    for k in range(TESTS_IN_PACK):
        d = {(True, True): 0,
             (True, False): 0,
             (False, True): 0,
             (False, False): 0}
        good = 0
        data_keeper.shuffle()
        tree = spatial.KDTree(data_keeper.learn_part)
        for (i, p) in enumerate(data_keeper.test_part):
            r = predict(tree, data_keeper, p, kernel, n)
            d[bool(r == data_keeper.test_value(i)), bool(r)] += 1

        print("Try #", k, '  |  accuracy:', '%5.3f' % Qualities.accuracy(d), ' | F-measure: ',
              '%5.3f' % Qualities.fMeasure(d))
        r_a += Qualities.accuracy(d)
        r_f += Qualities.fMeasure(d)

    print("Average:",' accuracy:', '%5.3f' % (r_a / 5.0), ' | F-measure: ',
          '%5.3f' % (r_f / 5.0))

    # dt.append({'accuracy': Qualities.accuracy(d), 'f-measure': Qualities.fMeasure(d)})

    # print("Average:", ans / TESTS_IN_PACK, "%")
    return dt


def main():
    data_keeper = DataKeeper(read_data())
    kernels = {
        'Gauss, sigma = 2': partial(Kernels.gaussian, sigma=2),
        'Gauss, sigma = 500': partial(Kernels.gaussian, sigma=500),
        'Gauss, sigma = 2000': partial(Kernels.gaussian, sigma=2000),
        'Squared': Kernels.squared}
    neighbor = range(2, 9)
    dt = pd.DataFrame()

    for n in neighbor:
        for ker_name, ker in kernels.items():
            print("________")
            print("Neighbors:", n, "Ker: ", ker_name)
            print('---')
            x = run_one(data_keeper, ker, n)
            dt.append(pd.DataFrame(x, index=['accuracy', 'f-measure']))
            # print(dt.to_clipboard())


if __name__ == '__main__':
    main()
