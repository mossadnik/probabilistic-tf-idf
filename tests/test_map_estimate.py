"""Tests for ptfidf.train.inference."""

import numpy as np


from ptfidf.inference import map_estimate
from ptfidf.aggregation import TokenStatistics


def test_map_estimate_easy():
    """
    check trivial example without matches.

    pi is population average,
    s stays at prior mean.
    """
    n = np.array([1, 1])
    k = np.array([0, 1])

    weights = np.array([
        [7, 3],
        [99, 1],
    ])

    prior_mean = -1.

    expected_pi = np.array([.3, .01])
    expected_s = np.exp(prior_mean) * np.ones_like(expected_pi)

    beta_params = map_estimate(TokenStatistics(n, k, weights), prior_mean, 1.)

    assert np.allclose(beta_params.frequency, expected_pi)
    assert np.allclose(beta_params.strength, expected_s)


def test_map_estimate_weight_deduplication():
    """Same as easy, but with duplicate weights to test deduplication.

    pi is population average,
    s stays at prior mean.
    """
    n = np.array([1, 1])
    k = np.array([0, 1])

    weights = np.array([
        [7, 3],
        [99, 1],
        [99, 1],
    ])

    prior_mean = -1.

    expected_pi = np.array([.3, .01, .01])
    expected_s = np.exp(prior_mean) * np.ones_like(expected_pi)

    beta_params = map_estimate(TokenStatistics(n, k, weights), prior_mean, 1.)

    assert np.allclose(beta_params.frequency, expected_pi)
    assert np.allclose(beta_params.strength, expected_s)
