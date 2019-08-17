"""Tests for ptfidf.train.inference."""

import numpy as np
from scipy.sparse import csr_matrix

from ptfidf.inference import map_estimate
from ptfidf import aggregation as agg


def test_map_estimate_easy():
    """
    check trivial example without matches.

    pi is population average,
    s stays at prior mean.
    """
    counts = np.array(
        [[1, 0]] * 7
        + [[0, 1]] * 3
    )
    nobs = np.ones(counts.shape[0], dtype=np.int32)

    token_stats = agg.TokenStatistics.from_entity_statistics(
        agg.EntityStatistics(csr_matrix(counts), nobs)
    )

    prior_mean = -1.

    expected_pi = np.array([.7, .3])
    expected_s = np.exp(prior_mean) * np.ones_like(expected_pi)

    beta_params = map_estimate(token_stats, prior_mean, 1.)

    assert np.allclose(beta_params.mean, expected_pi)
    assert np.allclose(beta_params.strength, expected_s)


def test_map_estimate_weight_deduplication():
    """Same as easy, but with duplicate weights to test deduplication.

    pi is population average,
    s stays at prior mean.
    """
    counts = np.array(
        [[1, 0, 1]] * 7
        + [[0, 1, 0]] * 3
    )
    nobs = np.ones(counts.shape[0], dtype=np.int32)

    token_stats = agg.TokenStatistics.from_entity_statistics(
        agg.EntityStatistics(csr_matrix(counts), nobs)
    )

    prior_mean = -1.

    expected_pi = np.array([.7, .3, .7])
    expected_s = np.exp(prior_mean) * np.ones_like(expected_pi)

    beta_params = map_estimate(token_stats, prior_mean, 1.)

    assert np.allclose(beta_params.mean, expected_pi)
    assert np.allclose(beta_params.strength, expected_s)
