import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ptfidf.train import aggregation


def test_group_statistics():
    """check group aggregation on small example."""
    X = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
    ])
    y = np.array([0, 1, 1, 2])

    expected_counts = np.array([
        [1, 0, 0],
        [1, 2, 0],
        [1, 0, 0]])
    expected_nobs = np.array([1, 2, 1])

    counts, nobs = aggregation.get_group_statistics(csr_matrix(X), y)

    assert np.all(expected_counts == counts.toarray()), 'counts do not match'
    assert np.all(nobs == expected_nobs), 'n_observations does not match'


def test_compress_group_statistics():
    """check aggregation on small example."""
    counts = np.array([
        [1, 1, 0],
        [1, 2, 0],
        [1, 0, 0]])
    nobs = np.array([1, 2, 1])

    expected_n = np.array([1, 1, 2, 2, 2])
    expected_k = np.array([0, 1, 0, 1, 2])
    expected_weights = np.array([
        [0, 2, 0, 1, 0],
        [1, 1, 0, 0, 1],
        [2, 0, 1, 0, 0]
    ])

    n, k, weights = aggregation.compress_group_statistics(csr_matrix(counts), nobs)

    assert np.all(n == expected_n)
    assert np.all(k == expected_k)
    assert np.allclose(weights, expected_weights)
