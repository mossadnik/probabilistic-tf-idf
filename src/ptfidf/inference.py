"""
MAP estimation
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from .likelihood import beta_binomial_log_likelihood, beta_binomial_log_likelihood_grad
from . import utils as ut


class NormalDist:
    """Container for parameters of Normal distribution.

    Parameters
    ----------
    mean : float or numpy.ndarray
    std: float or numpy.ndarray
    """

    __slots__ = ('mean', 'std')

    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __repr__(self):
        return 'NormalDist(mean={}, std={})'.format(
            ut.repr_maybe_array(self.mean),
            ut.repr_maybe_array(self.std)
        )


class BetaDist:
    """Container for parameters of Beta distribution.

    To initialize from mean / strength parametrization use
    `BetaParameters.from_mean_strength`

    Parameters
    ----------
    alpha : float or numpy.ndarray
    beta : float or numpy.ndarray
    """

    __slots__ = ('alpha', 'beta')

    def __init__(self, alpha, beta):
        """
        Use either (alpha, beta) or (mean, strength). If both are given,
        the first take precedence.
        """
        self.alpha = alpha
        self.beta = beta

    @classmethod
    def from_mean_strength(cls, mean, strength):
        """Instantiate using mean-strength parametrization.

        Parameters
        ----------
        mean : float or numpy.ndarray
        strength : float or numpy.ndarray
        """
        return cls(mean * strength, (1. - mean) * strength)

    def update(self, other, fraction=1.):
        """Update parameters."""
        for variable in ['alpha', 'beta']:
            ut.update(getattr(self, variable), getattr(other, variable), fraction)

    @property
    def mean(self):
        """get mean parameter"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength(self):
        """get strength parameter"""
        return self.alpha + self.beta

    def __repr__(self):
        return 'BetaDist(alpha={}, beta={})'.format(
            ut.repr_maybe_array(self.alpha),
            ut.repr_maybe_array(self.beta)
        )


def _pack(pi, s):
    """Convert params for optimizer."""
    return np.concatenate([logit(pi), np.log(s)])


def _unpack(x):
    """Convert optimizer vector to params."""
    n = x.size // 2
    pi = expit(x[:n])
    s = np.exp(x[n:])
    return pi, s


def _loss(x, positive_weights, negative_weights, total_weights, prior: NormalDist):
    pi, s = _unpack(x)
    alpha, beta = pi * s, (1 - pi) * s
    res = -beta_binomial_log_likelihood(
        alpha, beta,
        positive_weights, negative_weights, total_weights
    )
    # prior
    res += .5 * (np.log(s) - prior.mean)**2 / prior.std**2
    return res.sum()


def _loss_grad(x, positive_weights, negative_weights, total_weights, prior: NormalDist):
    pi, s = _unpack(x)
    # gradient for alpha, beta
    alpha, beta = pi * s, (1 - pi) * s
    grad_ab = -beta_binomial_log_likelihood_grad(
        alpha, beta,
        positive_weights, negative_weights, total_weights
    )
    # transformations
    grad = np.empty_like(grad_ab)
    # pi / s
    grad[0] = s * (grad_ab[0] - grad_ab[1])
    grad[1] = pi * grad_ab[0] + (1 - pi) * grad_ab[1]
    # link functions
    grad[0] *= pi * (1 - pi)
    grad[1] *= s
    # prior
    grad[1] += (np.log(s) - prior.mean) / prior.std**2
    return grad.ravel()


def init_beta_binomial_proba(positive_weights, negative_weights, a0=0., b0=0.):
    """Initialize frequencies with smoothed empirical means."""
    counts = np.arange(positive_weights.shape[1])
    positive_counts = positive_weights.dot(counts)
    negative_counts = negative_weights.dot(counts)
    return (a0 + positive_counts) / (a0 + b0 + positive_counts + negative_counts)


def map_estimate(token_stats, prior, strength_init=None, mean_init=None):
    """Compute MAP estimate of token-level prior parameters.

    Parameters
    ----------
    token_stats : ptfidf.aggregation.TokenStatistics
        Token-level statistics.
    prior : ptfidf.inference.NormalDist
        Prior distribution of log strength parameter.
    strength_init : numpy.ndarray, optional
        Initial value for strength parameter. Defaults to prior mean.
    mean_init : numpy.ndarray, optional
        Initial value for mean parameter. Defaults to a heuristic based
        on the strength parameter and token_stats.

    Returns
    -------
    ptfidf.inference.BetaParameters
        Estimated parameters.
    """
    # unique weights for saving multiple computation
    total_weights = token_stats.total_weights
    positive_weights, negative_weights, index, inverse, _ = token_stats.get_unique_weights()

    s = strength_init
    if s is None:
        s = np.exp(prior.mean) * np.ones(token_stats.size)
    pi = mean_init
    if pi is None:
        pi = init_beta_binomial_proba(positive_weights, negative_weights)

    res = minimize(
        _loss,
        _pack(pi[index], s[index]),
        args=(positive_weights, negative_weights, total_weights, prior),
        jac=_loss_grad,
        method='L-BFGS-B')

    if not res.success:
        raise RuntimeError('Optimization failed to converge.')
    pi, s = _unpack(res.x)
    return BetaDist.from_mean_strength(pi[inverse], s[inverse])
