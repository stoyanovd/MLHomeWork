import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import lena


def old_ex():
    img = lena()
    # the underlying signal is a sinusoidally modulated image
    t = np.arange(100)
    time = np.sin(0.1 * t)
    real = time[:, np.newaxis, np.newaxis] * img[np.newaxis, ...]

    # we add some noise
    noisy = real + np.random.randn(*real.shape) * 255

    # (observations, features) matrix
    M = noisy.reshape(noisy.shape[0], -1)


def read_data():
    return np.genfromtxt(os.path.join('..', 'data', 'Practice_PCA', 'newBasis3'), delimiter=' ')


def main():
    M = read_data()
    print(M.shape)

    # singular value decomposition factorises your data matrix such that:
    #
    #   M = U*S*V.T     (where '*' is matrix multiplication)
    #
    # * U and V are the singular matrices, containing orthogonal vectors of
    #   unit length in their rows and columns respectively.
    #
    # * S is a diagonal matrix containing the singular values of M - these
    #   values squared divided by the number of observations will give the
    #   variance explained by each PC.
    #
    # * if M is considered to be an (observations, features) matrix, the PCs
    #   themselves would correspond to the rows of S^(1/2)*V.T. if M is
    #   (features, observations) then the PCs would be the columns of
    #   U*S^(1/2).
    #
    # * since U and V both contain orthonormal vectors, U*V.T is equivalent
    #   to a whitened version of M.

    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    V = Vt.T

    # PCs are already sorted by descending order
    # of the singular values (i.e. by the
    # proportion of total variance they explain)

    # if we use all of the PCs we can reconstruct the noisy signal perfectly
    S = np.diag(s)

    def e_M(i):
        # if we use only the first 20 PCs the reconstruction is less accurate
        Mhat2 = np.dot(U[:, :i], np.dot(S[:i, :i], V[:, :i].T))
        err = np.mean((M - Mhat2) ** 2)
        print('Using first %d PCs, MSE = %.6G, ('"eigenvector : %.6G" % (i, err, S[i, i]))
        return err

    err = e_M(1)
    m = 1
    for m in range(2, 200):
        ne = e_M(m)
        if ne < 1e-2 and err / ne > 1e2:
            print('for', m, 'components', ne)
            break


if __name__ == '__main__':
    main()
