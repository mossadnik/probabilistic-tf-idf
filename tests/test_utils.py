# pylint: disable = redefined-outer-name
import pytest
import numpy as np
from scipy.sparse import csr_matrix

import ptfidf.utils as ut

@pytest.fixture
def unnormalized_probas():
    proba = np.array([
        [2., 1.00001, 0., 0.],
        [0.01, 2., 1.00001, 0.5],
        [0., 0., 0., 0.],
    ])

    prior = np.array([1., 3., 6.])

    log_proba = np.zeros_like(proba)
    idx = np.where(proba > 0)
    log_proba[idx] = np.log(proba[idx])
    log_proba = csr_matrix(log_proba)

    log_prior = np.zeros_like(prior)
    idx = np.where(prior > 0)
    log_prior[idx] = np.log(prior[idx])

    return proba, prior, log_proba, log_prior


def test_get_normalized_proba_with_prior(unnormalized_probas):
    proba, prior, log_proba, log_prior = unnormalized_probas
    expected = np.c_[proba, prior]
    expected /= expected.sum(axis=1, keepdims=True)
    expected = expected[:, :-1]

    actual = ut.get_normalized_proba(log_proba, log_prior).toarray()

    assert np.allclose(actual, expected)


def test_get_normalized_proba_without_prior(unnormalized_probas):
    proba, _, log_proba, _ = unnormalized_probas
    idx = np.sum(proba, axis=1) > 0
    expected = np.zeros_like(proba)
    expected[idx] = proba[idx] / proba[idx].sum(axis=1, keepdims=True)
    actual = ut.get_normalized_proba(log_proba).toarray()

    assert np.allclose(actual, expected)
