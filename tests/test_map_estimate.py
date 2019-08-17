"""Tests for ptfidf.train.inference."""
import pytest

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


@pytest.mark.parametrize('all_same,expect_less_than', [
    (1, True),
    (0, False)
])
def test_map_estimate_strength_trend(all_same, expect_less_than):
    """Test that strength changes into the correct direction.

    For cases that have two observations, the likelihood is maximized
    either at minus infinity (k=2, n=2) or plus infinity (k=1, n=2).

    Hence the optimized result should be less than / greater than the prior
    mean. When increasing the weights, they move away more.
    """

    prior_args = 0., 1.

    actual = []
    for w in range(1, 3):
        token_stats = agg.TokenStatistics(
            np.array([100, w]),
            np.array([[
                [10 + (1 - all_same) * w, all_same * w],
                [90 + (1 - all_same) * w, 0],
            ]])
        )
        params = map_estimate(token_stats, *prior_args)
        actual.append(np.log(params.strength[0]))

    # strength moves in correct direction
    assert (actual[0] < prior_args[0]) == expect_less_than
    # move further as evidence added
    assert (actual[1] < actual[0]) == expect_less_than
