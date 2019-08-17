"""Beta-Binomial likelihood and gradient."""

import numpy as np


def cum_log_gamma_ratio(x, n, deriv=0):
    """Evaluate a_i = sum_{j=0}^{i} log(x + j)} for i in [0, n - 1]."""
    arg = x[:, None] + np.arange(n)[None, :]
    if deriv == 0:
        res = np.log(arg)
    elif deriv == 1:
        res = 1. / arg
    elif deriv == 2:
        res = -arg**(-2.)
    else:
        raise NotImplementedError('Only derivatives up to second order implemented.')
    return np.cumsum(res, axis=-1)


def beta_binomial_log_likelihood(alpha, beta, weights, weights_n):
    """Compute log-likelihood for Beta-Binomial parameters."""
    n_max = weights_n.size

    res = np.sum(weights[:, 0] * cum_log_gamma_ratio(alpha, n_max), axis=-1)
    res += np.sum(weights[:, 1] * cum_log_gamma_ratio(beta, n_max), axis=-1)
    res -= cum_log_gamma_ratio(alpha + beta, n_max).dot(weights_n)
    return res


def beta_binomial_log_likelihood_grad(alpha, beta, weights, weights_n):
    """Compute gradient of log-likelihood for Beta-Binomial parameters."""
    kw = dict(deriv=1, n=weights_n.size)

    res = np.empty((2, alpha.size))
    res[0] = np.sum(weights[:, 0] * cum_log_gamma_ratio(alpha, **kw), axis=-1)
    res[1] = np.sum(weights[:, 1] * cum_log_gamma_ratio(beta, **kw), axis=-1)
    res -= cum_log_gamma_ratio(alpha + beta, **kw).dot(weights_n)[None, :]
    return res
