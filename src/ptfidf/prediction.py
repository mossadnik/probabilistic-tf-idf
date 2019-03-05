import numpy as np

from . import core


class BetaBernoulliModel(object):
    """Beta-Bernoulli observation model for sparse binary vectors."""
    def __init__(self, entities, token_prior, log_odds_matched=0.):
        """
        Parameters
        ----------
        entities : EntityStatistics
            Sufficient statistics of mixture components / classes.
        token_prior : BetaParameters
            Frequency / mean component of token-level Beta prior
        log_odds_matched : float, default 0.
            Log odds for assignment to existing entity rather than new one.
        """
        self.entities = entities
        self.token_prior = token_prior
        self.log_odds_matched = log_odds_matched

    def __repr__(self):
        s = 'BetaBinomialModel()'
        return s

    def get_log_proba(self, observations):
        """Get unnormalized observation log-probabilities."""
        return core.get_log_proba(observations, self.entities, self.token_prior.frequency, self.token_prior.strength)

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
        log_proba = self.get_log_proba(observations)
        log_prior = core.get_log_prior(observations, self.token_prior.frequency)
        return core.get_proba(log_proba, log_prior, self.log_odds_matched)

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
            res[i, :] = core.sample_assignments(proba)
        return res
