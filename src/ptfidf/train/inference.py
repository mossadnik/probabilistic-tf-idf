"""
MAP estimation
"""

import warnings

import numpy as np
from scipy.optimize import minimize

import autograd.numpy as anp
from autograd import grad

from .likelihood import beta_binomial_log_likelihood


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
    res += .5 * (anp.log(s) - prior_mean)**2 / prior_std**2
    return res.mean()


loss_grad = grad(loss)


def initialize_pi(n, k, weights, s):
    """guess beta-binomial optimal frequency given strength.

    Loosely based on Sec. 4.1 in
    https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf
    """
    a = 1. / (1. + 1. / s)  # interpolate count damping
    return weights.dot(k**a) / weights.dot(k**a + (n - k)**a)


def map_estimate(n, k, weights, prior_mean, prior_std):
    s = np.exp(prior_mean)
    pi = initialize_pi(n, k, weights, s)

    res = minimize(
        loss,
        _pack(pi, s * np.ones_like(pi)),
        args=(n, k, weights, prior_mean, prior_std),
        jac=loss_grad,
        method='L-BFGS-B')

    if not res.success:
        warnings.warn('failed to converge')
    return _unpack(res.x)
