"""
MAP estimation
"""

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from .likelihood import beta_binomial_log_likelihood, grad_beta_binomial_log_likelihood
from .utils import update


def _pack(pi, s):
    return np.concatenate([logit(pi), np.log(s)])


def _unpack(x):
    n = x.size // 2
    pi = expit(x[:n])
    s = np.exp(x[n:])
    return pi, s


def _loss(x, n, k, weights, multiplicities, prior_mean, prior_std):
    pi, s = _unpack(x)
    alpha, beta = pi * s, (1 - pi) * s
    res = -beta_binomial_log_likelihood(alpha[:, None], beta[:, None], k[None, :], n[None, :])
    res = np.sum(res * weights, axis=1)
    # prior
    res += .5 * (np.log(s) - prior_mean)**2 / prior_std**2
    # weight multiplicity
    res *= multiplicities
    return res.sum()


def _loss_grad(x, n, k, weights, multiplicities, prior_mean, prior_std):
    pi, s = _unpack(x)
    alpha, beta = pi * s, (1 - pi) * s
    grad_ab = -grad_beta_binomial_log_likelihood(alpha[:, None], beta[:, None], k[None, :], n[None, :])
    grad_ab = np.sum(grad_ab * weights, axis=-1)
    grad = np.empty_like(grad_ab)
    # pi / s
    grad[0] = s * (grad_ab[0] - grad_ab[1])
    grad[1] = pi * grad_ab[0] + (1 - pi) * grad_ab[1]
    # link functions
    grad[0] *= pi * (1 - pi)
    grad[1] *= s
    # prior
    grad[1] += (np.log(s) - prior_mean) / prior_std**2
    # weight multiplicity
    grad *= multiplicities[None, :]
    return grad.ravel()


def _initialize_pi(token_stats, strength):
    """Guess beta-binomial optimal mean parameter given strength.

    Loosely based on Sec. 4.1 in
    https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf
    """
    n, k, w = token_stats.n, token_stats.k, token_stats.weights
    a = 1. / (1. + 1. / strength[:, None])  # interpolate count damping
    return np.sum(w * k**a, axis=1) / np.sum(w * (k**a + (n - k)**a), axis=1)


def map_estimate(token_stats, prior_mean, prior_std, s_init=None, pi_init=None):
    """Compute MAP estimate of token-level prior parameters.

    Parameters
    ----------
    token_stats : ptfidf.aggregation.TokenStatistics
        Token-level statistics.
    prior_mean : float
        Prior mean of log(s). s has a log-normal prior
        distribution.
    prior_std : float
        Prior standard deviation of log(s).
    s_init : numpy.ndarray, optional
        Initial value for strength parameter. Defaults to prior mean.
    pi_init : numpy.ndarray, optional
        Initial value for mean parameter. Defaults to a heuristic based
        on the strength parameter and token_stats.

    Returns
    -------
    ptfidf.inference.BetaParameters
        Estimated parameters.
    """
    # unique weights for saving multiple computation
    weights, index, inverse, multiplicities = np.unique(
        token_stats.weights,
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True)
    s = np.exp(prior_mean) * np.ones(token_stats.size) if s_init is None else s_init
    pi = _initialize_pi(token_stats, s) if pi_init is None else pi_init

    res = minimize(
        _loss,
        _pack(pi[index], s[index]),
        args=(token_stats.n, token_stats.k, weights, multiplicities, prior_mean, prior_std),
        jac=_loss_grad,
        method='L-BFGS-B')

    if not res.success:
        warnings.warn('failed to converge')
    pi, s = _unpack(res.x)
    return BetaParameters(mean=pi[inverse], strength=s[inverse])


class BetaParameters:
    """Container for parameters of Beta distribution."""
    def __init__(self, alpha=None, beta=None, mean=None, strength=None):
        """
        Use either (alpha, beta) or (mean, strength). If both are given,
        the first take precedence.
        """
        if alpha is not None and beta is not None:
            self.alpha = alpha
            self.beta = beta
        else:
            self.alpha = strength * mean
            self.beta = strength * (1 - mean)

    def update(self, other, fraction=1.):
        """Update parameters."""
        for p in ['alpha', 'beta']:
            update(getattr(self, p), getattr(other, p), fraction)

    @property
    def mean(self):
        """get mean parameter"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength(self):
        """get strength parameter"""
        return self.alpha + self.beta
