import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import gammaln

from ptfidf.core import get_log_proba


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


def test_log_proba_values():
    """test that sparse log-proba computation reproduces direct implementation."""
    # inputs
    U = np.array([
        [1., 1., 0.],
        [0., 1., 1.]
    ])

    V = np.eye(3)
    n_obs = np.array([1, 2, 3])

    pi = np.array([.01, .001, .02])
    s = np.array([.1, 1., .1])

    # expected result: log-likelihood if at least one token matches, else zero
    alpha, beta = s * pi, s * (1 - pi)
    expected = np.zeros((U.shape[0], V.shape[0]))
    for i in range(U.shape[0]):
        for j in range(V.shape[0]):
            u, v, n = U[i], V[j], n_obs[j]
            if u.dot(v) > 0:
                expected[i, j] = expected_log_proba(u, v, n, alpha, beta)

    observed = get_log_proba(csr_matrix(U), csr_matrix(V), n_obs, pi, s).toarray()
    assert np.allclose(expected, observed)
