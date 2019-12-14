"""Beta-Binomial likelihood and gradient."""


import numpy as np
from scipy.special import gammaln, digamma


def sparse_gammaln_ratio(x, weights, deriv=0):
    """Compute gammaln(x + n) - gammaln(x) parametrized with sparse weights."""
    if deriv == 0:
        func = gammaln
    elif deriv == 1:
        func = digamma
    else:
        raise NotImplementedError('Only derivatives up to first order supported')
    res = weights.tocoo(copy=True)
    x_row = x[res.row]
    res.data = func(x_row + res.col) - func(x_row)
    return res.multiply(weights).sum(axis=1).A.ravel()


def dense_gammaln_ratio(x, weights, deriv=0):
    """Compute gammaln(x + n) - gammaln(x) parametrized with dense weights."""
    if deriv == 0:
        func = gammaln
    elif deriv == 1:
        func = digamma
    else:
        raise NotImplementedError('Only derivatives up to first order supported')
    counts = np.where(weights)[0]
    weights = weights[counts]
    print(counts, weights, x, func(x[:, None] + counts[None, :]), func(x))
    return func(x[:, None] + counts[None, :]).dot(weights) - func(x) * weights.sum()


def beta_binomial_log_likelihood(
        alpha, beta,
        positive_weights, negative_weights, total_weights
):
    """Beta-binomial likelihood"""
    res = sparse_gammaln_ratio(alpha, positive_weights)
    res += sparse_gammaln_ratio(beta, negative_weights)
    res -= dense_gammaln_ratio(alpha + beta, total_weights)
    return res


def beta_binomial_log_likelihood_grad(
        alpha, beta,
        positive_weights, negative_weights, total_weights
):
    """Gradient of Beta-Binomial likelihood"""
    res = np.empty((2, alpha.size))
    res[0] = sparse_gammaln_ratio(alpha, positive_weights, deriv=1)
    res[1] = sparse_gammaln_ratio(beta, negative_weights, deriv=1)
    res -= dense_gammaln_ratio(alpha + beta, total_weights, deriv=1)
    return res
