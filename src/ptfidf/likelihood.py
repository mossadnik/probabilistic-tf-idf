"""Beta-Binomial likelihood and gradient."""

import numpy as np
from scipy.special import gammaln, digamma


def beta_log_normalizer(alpha, beta):
    """Log-normalizer of Beta distribution."""
    return gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)


def beta_binomial_log_likelihood(alpha, beta, positive, total):
    """Log-likelihood of Beta-Binomial distribution."""
    return (
        beta_log_normalizer(alpha + positive, beta + total - positive) -
        beta_log_normalizer(alpha, beta))


def grad_beta_log_normalizer(alpha, beta):
    res = np.empty((2,) + alpha.shape, dtype=alpha.dtype)
    res[0], res[1] = digamma(alpha), digamma(beta)
    res -= digamma(alpha + beta)[None, :]
    return res


def grad_beta_binomial_log_likelihood(alpha, beta, positive, total):
    return (
        grad_beta_log_normalizer(alpha + positive, beta + total - positive) -
        grad_beta_log_normalizer(alpha, beta))
