"""Low-level observation model functions."""

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from .signals import Observer, check_uptodate
from . import utils as ut


class SparseBetaBernoulliModel(Observer):
    def __init__(self, entities, prior):
        super().__init__()
        self.entities = entities
        self.prior = prior
        self.prior.subscribe(self)
        self.entities.subscribe(self)

    def update(self):
        """Get observation-independent data structures for observation likelihood."""
        self.uptodate = True
        counts, n_observations = self.entities.counts, self.entities.n_observations
        alpha, beta = self.prior.alpha, self.prior.beta

        # compute p^0_t for all relevant n
        max_observations = n_observations.max()
        n = np.arange(max_observations + 1, dtype=np.float32)[None, :]
        p0 = alpha[:, None] / (alpha[:, None] + beta[:, None] + n)

        # count-independent term
        unconstrained_term = np.log(1. - p0).sum(axis=0)

        # count-dependent terms
        k = counts.data  # count vectors
        n = n_observations[ut.sparse_row_indices(counts)]  # observation numbers
        t = counts.indices  # token indices

        t_in_k_cap_x_term = csr_matrix(
            (np.log((beta[t] + n) / (beta[t] + n - k)), counts.indices, counts.indptr),
            shape=counts.shape)
        t_in_k_term = -np.array(t_in_k_cap_x_term.sum(axis=1)).ravel()  # note the minus
        t_in_k_cap_x_term.data += np.log((alpha[t] + k) / alpha[t])

        self._p0_log_odds = np.log(p0 / (1. - p0))
        self._t_in_k_cap_x_term = csc_matrix(t_in_k_cap_x_term.T)
        self._t_in_k_term = t_in_k_term
        self._unconstrained_term = unconstrained_term

    @check_uptodate
    def get_log_proba(self, observations):
        """Compute log observation probabilites of observations."""
        # \sum_{t \in x \cap k}
        log_proba = observations.dot(self._t_in_k_cap_x_term)
        # \sum_{t \in k}
        log_proba.data += self._t_in_k_term[log_proba.indices]
        # \sum_{t \in x}
        n = self.entities.n_observations[log_proba.indices]
        t_in_x_term = observations.dot(self._p0_log_odds)
        row = ut.sparse_row_indices(log_proba)
        log_proba.data += t_in_x_term[row, n]
        # \sum_t
        log_proba.data += self._unconstrained_term[n]

        return log_proba

    @check_uptodate
    def get_log_prior(self, observations):
        """Compute log-prior of observations.

        Parameters
        ----------
        observations : (N, T) scipy.sparse.csr_matrix
            binary document-term matrix

        Returns
        -------
        prior : numpy.ndarray
            log-prior for each observation
        """
        alpha, beta = self.prior.alpha, self.prior.beta
        prior = observations.dot(np.log(alpha / beta))
        prior += np.log(beta / (alpha + beta)).sum()
        return prior
