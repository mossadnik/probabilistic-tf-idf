"""Utilities for Beta-Binomial likelihood."""

from autograd.scipy.special import gammaln

from .utils import damped_update


def beta_log_normalizer(alpha, beta):
    """Log-normalizer of Beta distribution."""
    return gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)


def beta_binomial_log_likelihood(alpha, beta, positive, total):
    """Log-likelihood of Beta-Binomial distribution."""
    return (
        beta_log_normalizer(alpha + positive, beta + total - positive) -
        beta_log_normalizer(alpha, beta))
