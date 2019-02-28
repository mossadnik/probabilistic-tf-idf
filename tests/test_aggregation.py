"""Tests for ptfidf.train.aggregation"""

import numpy as np
from scipy.sparse import csr_matrix

from ptfidf.train import aggregation as agg


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

    actual = agg.get_entity_statistics(csr_matrix(X), y)

    assert np.all(expected_counts == actual.counts.toarray()), 'counts do not match'
    assert np.all(actual.n_observations == expected_nobs), 'n_observations does not match'


def test_compress_group_statistics():
    """check aggregation on small example."""
    counts = np.array([
        [1, 1, 0],
        [1, 2, 0],
        [1, 0, 0]])
    nobs = np.array([1, 2, 1])
    entity_stats = agg.EntityStatistics(csr_matrix(counts), nobs)

    expected_n = np.array([1, 1, 2, 2, 2])
    expected_k = np.array([0, 1, 0, 1, 2])
    expected_weights = np.array([
        [0, 2, 0, 1, 0],
        [1, 1, 0, 0, 1],
        [2, 0, 1, 0, 0]
    ])

    token_stats = agg.get_token_statistics(entity_stats)

    assert np.all(token_stats.n == expected_n)
    assert np.all(token_stats.k == expected_k)
    assert np.allclose(token_stats.weights, expected_weights)


def test_idx2nk():
    """Test conversion between (n, k) and integer index."""
    n = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3])
    k = np.array([0, 1, 0, 1, 2, 0, 1, 2, 3])
    expected_idx = np.arange(n.size)

    idx = agg._nk2idx(n, k)
    assert np.all(idx == expected_idx)

    n_actual, k_actual = agg._idx2nk(idx)
    assert np.all(n_actual == n)
    assert np.all(k_actual == k)


def test_add_token_statisitics_index():
    """Test merging when indices differ."""
    left = agg.TokenStatistics(
        np.array([1, 1]),
        np.array([0, 1]),
        np.array([[1, 2]]))

    right = agg.TokenStatistics(
        np.array([1, 1, 2, 2, 3]),
        np.array([0, 1, 0, 1, 1]),
        np.array([[3, 5, 8, 9, 10]]))

    expected_n = np.array([1, 1, 2, 2, 3])
    expected_k = np.array([0, 1, 0, 1, 1])
    expected_weights = np.array([[4, 7, 8, 9, 10]])

    left.add(right)
    assert np.all(left.n == expected_n)
    assert np.all(left.k == expected_k)
    assert np.all(left.weights == expected_weights)
