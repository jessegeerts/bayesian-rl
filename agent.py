import numpy as np


class BayesQlearner(object):
    """Use bayesian Q learning to estimate action values with confidence.

    At some point, we're gonna have to estimate the beta parameters given new mean and variance.

    R code for doing so:

    estBetaParams <- function(mu, var) {
        alpha <- ((1 - mu) / var - 1 / mu) * mu ^ 2
        beta <- alpha * (1 / mu - 1)
        return(params = list(alpha = alpha, beta = beta))
    }

    """

    def __init__(self, environment):
        pass

