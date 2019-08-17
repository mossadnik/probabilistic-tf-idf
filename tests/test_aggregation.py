"""Tests for ptfidf.train.aggregation"""

import numpy as np
from scipy.sparse import csr_matrix

from ptfidf import aggregation as agg


def test_entity_statistics_from_observations():
    """check entity aggregation on small example."""
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

    actual = agg.EntityStatistics.from_observations(csr_matrix(X), y)

    assert np.all(expected_counts == actual.counts.toarray())
    assert np.all(actual.n_observations == expected_nobs)


def test_token_statistics_from_observations():
    observations = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ])

    expected_max_count = 1
    expected_pos_weights = np.sum(observations, axis=0).reshape((-1, 1))
    expected_neg_weights = np.sum(1 - observations, axis=0).reshape((-1, 1))

    token_stats = agg.TokenStatistics.from_observations(csr_matrix(observations))

    assert token_stats.max_count == expected_max_count
    assert np.all(token_stats.weights[:, 0, :] == expected_pos_weights)
    assert np.all(token_stats.weights[:, 1, :] == expected_neg_weights)


def test_token_statistics_from_entity_statistics():
    """Test aggregation for basic case."""
    counts = np.array([
        [1, 1, 0],
        [1, 2, 0],
        [1, 0, 0]])
    nobs = np.array([1, 3, 1])
    entity_stats = agg.EntityStatistics(csr_matrix(counts), nobs)

    expected_max_count = entity_stats.n_observations.max()
    # weights_n contains counts of entity_stats.n_observations
    expected_n_weights = np.array([2, 0, 1])

    # counts of
    #   entity_stats.counts
    # per token
    expected_pos_counts = np.array([
        [3, 0, 0],
        [1, 1, 0],
        [0, 0, 0],
    ])

    # counts of
    #   entity_stats.n_observations[:, None] - entity_stats.counts()
    # per token
    expected_neg_counts = np.array([
        [0, 1, 0],
        [2, 0, 0],
        [2, 0, 1],
    ])

    token_stats = agg.TokenStatistics.from_entity_statistics(entity_stats)

    assert token_stats.max_count == expected_max_count
    assert np.all(token_stats.weights_n == expected_n_weights)
    assert np.all(token_stats.weights[:, 0, :] == expected_pos_counts)
    assert np.all(token_stats.weights[:, 1, :] == expected_neg_counts)


def test_add_token_statistics():
    """Test adding TokenStatistics gives the same result as concatenating entities."""
    left_counts = np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 2]
    ])
    left_nobs = np.array([1, 2])
    left_entity_stats = agg.EntityStatistics(csr_matrix(left_counts), left_nobs)

    right_counts = np.array([
        [0, 1, 0, 1],
        [3, 0, 0, 0],
    ])
    right_nobs = np.array([1, 3])
    right_entity_stats = agg.EntityStatistics(csr_matrix(right_counts), right_nobs)

    summed_entity_stats = agg.EntityStatistics(
        csr_matrix(np.concatenate([left_counts, right_counts], axis=0)),
        np.concatenate([left_nobs, right_nobs])
    )
    expected_token_stats = agg.TokenStatistics.from_entity_statistics(summed_entity_stats)

    left_token_stats = agg.TokenStatistics.from_entity_statistics(left_entity_stats)
    right_token_stats = agg.TokenStatistics.from_entity_statistics(right_entity_stats)
    actual_token_stats = left_token_stats.add(right_token_stats)

    assert np.all(expected_token_stats.weights_n == actual_token_stats.weights_n)
    assert np.all(expected_token_stats.weights == actual_token_stats.weights)
