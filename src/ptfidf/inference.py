"""
MAP estimation
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from .likelihood import beta_binomial_log_likelihood, beta_binomial_log_likelihood_grad
from .utils import update


def _pack(pi, s):
    """Convert params for optimizer."""
    return np.concatenate([logit(pi), np.log(s)])


def _unpack(x):
    """Convert optimizer vector to params."""
    n = x.size // 2
    pi = expit(x[:n])
    s = np.exp(x[n:])
    return pi, s


def _loss(x, weights, weights_n, prior_mean, prior_std):
    pi, s = _unpack(x)
    alpha, beta = pi * s, (1 - pi) * s
    res = -beta_binomial_log_likelihood(alpha, beta, weights, weights_n)
    # prior
    res += .5 * (np.log(s) - prior_mean)**2 / prior_std**2
    return res.sum()


def _loss_grad(x, weights, weights_n, prior_mean, prior_std):
    pi, s = _unpack(x)
    # gradient for alpha, beta
    alpha, beta = pi * s, (1 - pi) * s
    grad_ab = -beta_binomial_log_likelihood_grad(alpha, beta, weights, weights_n)
    # transformations
    grad = np.empty_like(grad_ab)
    # pi / s
    grad[0] = s * (grad_ab[0] - grad_ab[1])
    grad[1] = pi * grad_ab[0] + (1 - pi) * grad_ab[1]
    # link functions
    grad[0] *= pi * (1 - pi)
    grad[1] *= s
    # prior
    grad[1] += (np.log(s) - prior_mean) / prior_std**2
    return grad.ravel()


def init_beta_binomial_proba(weights, a0=0., b0=0.):
    """Initialize frequencies with smoothed empirical means."""
    n_max = weights.shape[-1]
    counts = np.sum(weights * np.arange(1, n_max + 1)[None, None, :], axis=-1)
    return (a0 + counts[:, 0]) / (a0 + b0 + counts.sum(axis=1))


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
    weights_n = token_stats.weights_n
    weights, index, inverse = np.unique(
        token_stats.weights,
        axis=0,
        return_index=True,
        return_inverse=True
    )
    s = np.exp(prior_mean) * np.ones(token_stats.size) if s_init is None else s_init
    pi = init_beta_binomial_proba(token_stats.weights) if pi_init is None else pi_init

    res = minimize(
        _loss,
        _pack(pi[index], s[index]),
        args=(weights, weights_n, prior_mean, prior_std),
        jac=_loss_grad,
        method='L-BFGS-B')

    if not res.success:
        raise RuntimeError('Optimization failed to converge.')
    pi, s = _unpack(res.x)
    return BetaDist.from_mean_strength(pi[inverse], s[inverse])


class BetaDist:
    """Container for parameters of Beta distribution.

    To initialize from mean / strength parametrization use
    `BetaParameters.from_mean_strength`

    Parameters
    ----------
    alpha : float or numpy.ndarray
    beta : float or numpy.ndarray
    """
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
            update(getattr(self, variable), getattr(other, variable), fraction)

    @property
    def mean(self):
        """get mean parameter"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength(self):
        """get strength parameter"""
        return self.alpha + self.beta
