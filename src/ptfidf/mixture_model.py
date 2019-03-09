"""Mixture / partitioning"""

import numpy as np
from scipy.sparse import csr_matrix

from . import utils as ut


def _get_proba(log_proba, log_prior, log_odds=0.):
    """Compute normalized match probabilities.

    Takes into account possibility of creating a new partition,
    i.e. name not in list.

    Parameters
    ----------
    log_proba : scipy.sparse.csr_matrix
        unnormalized log-probabilities. Observations in rows,
        classes in columns.
    log_prior : numpy.ndarray
        log-prior for each observation. Size must match number
        of rows of log_proba.
    log_odds : float, optional
        log-odds for assignment to an existing class vs creating a new one.

    Returns
    -------
    proba : scipy.sparse.csr_matrix
        normalized probabilities. Same shape and sparsity structure as
        log_proba. Sum over rows is the probability to belong to any
        existing class (Rows sum to less than one).
    """
    proba = csr_matrix(
        (np.empty_like(log_proba.data), log_proba.indices, log_proba.indptr),
        shape=log_proba.shape
    )

    for row, slc in ut.sparse_iter_rows(log_proba):
        lp = log_proba.data[slc] - log_prior[row] + log_odds
        lp_max = lp.max()
        lp -= lp_max
        norm = np.log(np.sum(np.exp(lp)) + np.exp(-lp_max))
        proba.data[slc] = np.exp(lp - norm)

    return proba


def sample_assignments(proba):
    """Randomly assing observations according to distribution.

    Parameters
    ----------
    proba : scipy.sparse.csr_matrix
        Assignment probabilities.

    Returns
    -------
    assignments : numpy.ndarray of int
        Sampled assignments, -1 is used to indicate
        unassigned (i.e. assigned to new entity).
    """
    n_test = proba.shape[0]
    assignments = -np.ones(n_test, dtype=np.int32)
    rnd = np.random.rand(n_test)
    for row, slc in ut.sparse_iter_rows(proba):
        choice = np.searchsorted(np.cumsum(proba.data[slc]), rnd[row])
        if choice < slc.stop - slc.start:
            assignments[row] = proba.indices[slc][choice]
    return assignments


class MixtureModel:
    """Beta-Bernoulli observation model for sparse binary vectors."""
    def __init__(self, observation_model, log_odds_matched=0.):
        """
        Parameters
        ----------
        entities : EntityStatistics
            Sufficient statistics of mixture components / classes.
        log_odds_matched : float, default 0.
            Log odds for assignment to existing entity rather than new one.
        """
        self.observation_model = observation_model
        self.log_odds_matched = log_odds_matched

    def __repr__(self):
        s = 'BetaBinomialModel()'
        return s

    # this actually applies the partition model on top of observations
    def get_proba(self, observations):
        """
        Get normalized observation probabilities.

        Takes into account possibility of creating a new partition / entity.

        Parameters
        ----------
        observations : scipy.sparse.csr_matrix
            Binary observation vectors stored in rows.

        Returns
        -------
        proba : scipy.sparse.csr_matrix
            Assignment probabilities for all observations. Probabilities sum to less than
            one in general, the missing mass is the probability to be assigned to a new
            entity.
        """
        return _get_proba(
            self.observation_model.get_log_proba(observations),
            self.observation_model.get_log_prior(observations),
            self.log_odds_matched)

    def sample_assignments(self, observations, n_sample=1):
        """
        Sample assignments of observations to entities from predictive distribution.

        Parameters
        ----------
        observations : scipy.sparse.csr_matrix
            Binary observation vectors stored in rows.
        n_sample : int, default 1
            number of samples.

        Returns
        -------
        samples : numpy.ndarray
            Sampled assignments. -1 means assigned to new entity.
            Shape (n_sample, observations.shape[0])
        """
        proba = self.get_proba(observations)
        res = np.empty((n_sample, observations.shape[0]), dtype=np.int32)
        for i in range(n_sample):
            res[i, :] = sample_assignments(proba)
        return res
