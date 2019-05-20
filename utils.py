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


def product_var(exp_x, exp_y, var_x, var_y):
    """Compute the variance of the product of two independent random variables given their expectation and variance.

    :param exp_x:
    :param exp_y:
    :param var_x:
    :param var_y:
    :return:
    """
    return var_x * var_y + var_x * exp_y**2 + var_y * exp_x**2


def dotproduct_var(exp_x, exp_y, var_x, var_y):
    """Compute the variance of the product of two independent random variables given their expectation and variance.

    :param exp_x:
    :param exp_y:
    :param var_x:
    :param var_y:
    :return:
    """
    var = sum([product_var(exp_x[i], exp_y[i], var_x[i], var_y[i]) for i in range(len(exp_x))])
    return var

