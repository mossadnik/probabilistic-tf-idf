"""
MAP estimation
"""

import warnings

import numpy as np
from scipy.optimize import minimize

import autograd.numpy as anp
from autograd import grad

from .likelihood import beta_binomial_log_likelihood, BetaParameters


def _pack(pi, s):
    return anp.concatenate([anp.log(pi / (1. - pi)), anp.log(s)])


def _unpack(x):
    n = x.size // 2
    pi = 1. / (1. + anp.exp(-x[:n]))
    s = anp.exp(x[n:])
    return pi, s


def loss(x, n, k, weights, prior_mean, prior_std):
    pi, s = _unpack(x)
    a, b = pi * s, (1 - pi) * s
    res = -beta_binomial_log_likelihood(a[:, None], b[:, None], k[None, :], n[None, :])
    res = anp.sum(res * weights, axis=1)
    # from here on, need to add in the weights
    res += .5 * (anp.log(s) - prior_mean)**2 / prior_std**2
    return res.mean()


loss_grad = grad(loss)


def _initialize_pi(token_stats, strength):
    """Guess beta-binomial optimal mean parameter given strength.

    Loosely based on Sec. 4.1 in
    https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf
    """
    # don't use token_stats in interface for flexibility
    n, k, w = token_stats.n, token_stats.k, token_stats.weights
    a = 1. / (1. + 1. / strength)  # interpolate count damping
    return w.dot(k**a) / w.dot(k**a + (n - k)**a)


def map_estimate(token_stats, prior_mean, prior_std, s_init=None, pi_init=None):
    # init: use first matching index, average or random
    s = np.exp(prior_mean) if s_init is None else s_init
    pi = _initialize_pi(token_stats, s) if pi_init is None else pi_init

    res = minimize(
        loss,
        _pack(pi, s * np.ones_like(pi)),
        args=(token_stats.n, token_stats.k, token_stats.weights, prior_mean, prior_std),
        jac=loss_grad,
        method='L-BFGS-B')

    if not res.success:
        warnings.warn('failed to converge')
    # postprocessing: map compressed parameters back
    return BetaParameters(*_unpack(res.x))
