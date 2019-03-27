import numpy as np


def exponentiate(M, exp):
    """Exponentiate a matrix element-wise. For a diagonal matrix, this is equivalent to matrix exponentiation.

    :param M:
    :param exp:
    :return:
    """
    num_rows = len(M)
    num_cols = len(M[0])
    exp_m = np.zeros((num_rows, num_cols))

    for i in range(num_rows):
        for j in range(num_cols):
            if M[i][j] != 0:
                exp_m[i][j] = M[i][j] ** exp

    return exp_m


def all_argmax(x):
    """Argmax operation, but if there are multiple maxima, return all.

    :param x: Input data
    :return: chosen index
    """
    x = np.array(x)
    arg_maxes = (x == x.max())
    indices = np.flatnonzero(arg_maxes)
    return indices


def softmax(x, beta=2):
    """Compute the softmax function.

    :param x: Data
    :param beta: Inverse temperature parameter.
    :return:
    """
    x = np.array(x)
    return np.exp(beta * x) / sum(np.exp(beta * x))
