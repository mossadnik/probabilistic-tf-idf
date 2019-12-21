"""Low-level observation model functions."""

import numpy as np
from scipy.sparse import csr_matrix


class SparseBetaBernoulliModel:
    """Beta-Bernoulli model for sparse binary vectors."""
    def __init__(self, entities, prior):
        self._entities = entities
        self._prior = prior
        self._initialize()

    @property
    def entities(self):
        """return entities."""
        return self._entities

    @property
    def prior(self):
        """return token-level prior."""
        return self._prior

    def _initialize(self):
        """Get observation-independent data structures for observation likelihood."""

        n_observations = self.entities.n_observations
        # ensure new buffer so that we can overwrite later
        counts = self.entities.counts.tocoo(copy=True)
        alpha, beta = self.prior.alpha, self.prior.beta

        # compute p^0_t for all relevant n
        max_observations = n_observations.max()
        n = np.arange(max_observations + 1, dtype=np.float32)[None, :]
        p_0 = alpha[:, None] / (alpha[:, None] + beta[:, None] + n)

        # count-independent term
        unconstrained_term = np.log(1. - p_0).sum(axis=0)

        # count-dependent terms
        k = counts.data  # count vectors
        n = n_observations[counts.row]  # observation numbers
        token_idx = counts.col  # token indices
        t_in_k_cap_x_term = counts  # reuse buffers with new name
        t_in_k_cap_x_term.data = np.log((beta[token_idx] + n) / (beta[token_idx] + n - k))
        t_in_k_term = -np.array(t_in_k_cap_x_term.sum(axis=1)).ravel()  # note the sign
        t_in_k_cap_x_term.data += np.log((alpha[token_idx] + k) / alpha[token_idx])

        self._p0_log_odds = np.log(p_0 / (1. - p_0))
        self._t_in_k_cap_x_term = csr_matrix(t_in_k_cap_x_term.T)
        self._t_in_k_term = t_in_k_term
        self._unconstrained_term = unconstrained_term

    def get_log_proba(self, observations):
        """Compute log observation probabilites of observations."""
        # \sum_{t \in x \cap k}
        log_proba = observations.dot(self._t_in_k_cap_x_term).tocoo()
        # \sum_{t \in k}
        log_proba.data += self._t_in_k_term[log_proba.col]
        # \sum_{t \in x}
        n = self.entities.n_observations[log_proba.col]
        t_in_x_term = observations.dot(self._p0_log_odds)
        log_proba.data += t_in_x_term[log_proba.row, n]
        # \sum_t
        log_proba.data += self._unconstrained_term[n]

        return log_proba.tocsr()

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
