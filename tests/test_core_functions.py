"""Tests for ptfidf.core."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import gammaln

from ptfidf.core import get_log_proba, SparseBetaBernoulliModel
from ptfidf.train.aggregation import EntityStatistics


def beta_log_normalizer(a, b):
    """log-normalizer of Beta distribution."""
    return gammaln(a) + gammaln(b) - gammaln(a + b)


def expected_log_proba(x, k, n, alpha, beta):
    """Beta-binomial conditional likelihood

    p(x | k, n, alpha, beta) = p(k + x, n + 1 | alpha, beta) / p(k, n | alpha, beta)

    with

    p(k, n | alpha, beta) = Z(alpha + k, beta + n - k) / Z(alpha, beta)

    and

    Z(alpha, beta) = Gamma(alpha) * Gamma(beta) / Gamma(alpha + beta)
    """
    a, b = alpha, beta
    return np.sum(
        beta_log_normalizer(a + x + k, b + n + 1 - k - x) -
        beta_log_normalizer(a + k, b + n - k))


def test_model_get_log_proba():
    """test that sparse log-proba computation reproduces direct implementation."""
    # inputs
    X = np.array([
        [1., 1., 0.],
        [0., 1., 1.]
    ])

    counts = np.eye(3)
    n_obs = np.array([1, 2, 3])

    pi = np.array([.01, .001, .02])
    s = np.array([.1, 1., .1])

    # expected result: log-likelihood if at least one token matches, else zero
    alpha, beta = s * pi, s * (1 - pi)
    expected = np.zeros((X.shape[0], counts.shape[0]))
    for i in range(X.shape[0]):
        for j in range(counts.shape[0]):
            u, v, n = X[i], counts[j], n_obs[j]
            if u.dot(v) > 0:
                expected[i, j] = expected_log_proba(u, v, n, alpha, beta)

    entity_stats = EntityStatistics(csr_matrix(counts), n_obs)
    model = SparseBetaBernoulliModel(entity_stats, pi, s)
    actual = model.get_log_proba(csr_matrix(X)).toarray()
    assert np.allclose(actual, expected)
