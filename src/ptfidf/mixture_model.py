"""Mixture / partitioning"""

from . import utils as ut


class MixtureModel:
    """Simple Mixture Model."""
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
        s = 'MixtureModel()'
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
        return ut.get_normalized_proba(
            self.observation_model.get_log_proba(observations),
            self.observation_model.get_log_prior(observations) - self.log_odds_matched)
