"""Containers and aggregation functions for sufficient statistics."""

import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

from .. import utils as ut


class EntityStatistics(object):
    """Container for entity-level sufficient statistics."""
    def __init__(self, counts, n_observations):
        self.counts = counts
        self.n_observations = n_observations

    @property
    def size(self):
        """Get number of entities."""
        return self.n_observations.size

    def __repr__(self):
        s = 'EntityStatistics({} entities, {} observations)'
        return s.format(self.counts.shape[0], self.n_observations.sum())

    def add(self, other):
        """add sufficient statistics."""
        if not other.counts.shape == self.counts.shape:
            raise ValueError('Incompatible shapes.')
        self.counts += other.counts
        self.n_observations += other.n_observations
        return self

    def copy(self):
        """return new EntityStatistics with copied parameter arrays."""
        return self.__class__(self.counts.copy(), self.n_observations.copy())


def get_entity_statistics(X, y, n_classes=None):
    """
    Aggregate observations to entity level.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        binary document-term matrix
    y : numpy.ndarray
        class labels. Must have y.size == X.shape[0]

    Returns
    -------
    counts : scipy.sparse.csr_matrix
        token count vectors on group level. Shape
        (max(y) + 1, X.shape[1]).
    n_observations : numpy.ndarray
        number of observations for each group. Shape
        n_observations.size == counts.shape[0].
    """
    if y.size != X.shape[0]:
        raise ValueError('Incompatible shapes.')
    n = y.size
    if n_classes is None:
        n_classes = y.max() + 1
    dt = np.int32
    data, row, col = np.ones(n, dtype=dt), y.astype(dt), np.arange(n, dtype=dt)
    assignments = csr_matrix((data, (row, col)), shape=(n_classes, X.shape[0]))
    counts = assignments.dot(X)
    n_observations = np.array(assignments.sum(axis=1)).ravel()

    return EntityStatistics(counts, n_observations)


def _nk2idx(n, k):
    """convert n, k to integer index."""
    return ((n - 1) * (n + 2)) // 2 + k


def _idx2nk(idx):
    """
    convert integer index to n, k.

    inverse of _nk2idx
    """
    n = np.floor(-.5 + .5 * np.sqrt(9 + 8 * idx)).astype(int)
    k = idx - (n - 1) * (n + 2) // 2
    return n, k


class TokenStatistics(object):
    """Container for token-level sufficient statistics."""
    def __init__(self, n, k, weights):
        self.n = n
        self.k = k
        self.weights = weights

    @property
    def size(self):
        """get number of tokens."""
        return self.weights.shape[0]

    def __repr__(self):
        s = 'TokenStatistics({} tokens)'
        return s.format(self.size)


    def add(self, other):
        """merge counts, add weights."""
        if self.size != other.size:
            raise ValueError('Shape mismatch: number of token differs.')
        idx_self = _nk2idx(self.n, self.k)
        idx_other = _nk2idx(other.n, other.k)
        idx = np.union1d(idx_self, idx_other)
        if idx_self.size == idx.size and np.all(idx == idx_self):
            self.weights[:, idx_other] += other.weights
        else:
            weights = np.zeros((self.weights.shape[0], idx.size))
            weights[:, np.searchsorted(idx, idx_self)] += self.weights
            weights[:, np.searchsorted(idx, idx_other)] += other.weights
            self.weights = weights
            self.n, self.k = _idx2nk(idx)
        return self

    def copy(self):
        """Return new TokenStatistics instance with copied parameter arrays."""
        return self.__class__(self.n.copy(), self.k.copy(), self.weights.copy())


def get_token_statistics(entity_stats):
    """
    Aggregate entity-level sufficient statistics into distinct likelihood terms.

    Parameters
    ----------
    entity_stats : EntityStatistics

    Returns
    -------
    pandas.DataFrame
        Sufficient statistics and weights for all tokens
        likelihood terms. Columns ['token', 'n', 'k', 'weight']
        with primary key ['token', 'n', 'k'].
    """
    n_tokens = entity_stats.counts.shape[1]
    distinct_nobs, count_nobs = np.unique(entity_stats.n_observations, return_counts=True)

    # Aggregate counts into distinct likelihood terms with weights / multiplicities
    res = ut.sparse_to_frame(entity_stats.counts).rename(columns={'row': 'group', 'col': 'token', 'data': 'k'})
    res['n'] = entity_stats.n_observations[res['group'].values]
    res = res.groupby(['token', 'n', 'k']).size().reset_index().rename(columns={0: 'weight'})

    # fill in missing groups with no token observations (k == 0):
    # Compute weight of terms (token, n, k == 0) by subtracting sum of weights
    # for all (token, n, k > 0) from number of groups with n observations
    not_observed = pd.DataFrame({
        k: arr.ravel() for k, arr in zip(['token', 'n'], np.meshgrid(np.arange(n_tokens), distinct_nobs))})
    not_observed['weight'] = not_observed['n'].map(pd.Series(index=distinct_nobs, data=count_nobs))
    not_observed.set_index(['token', 'n'], inplace=True)
    not_observed = (
        not_observed
        .subtract(res.groupby(['token', 'n'])[['weight']].sum(), fill_value=0)
        .assign(k=0)
        .reset_index()
        .loc[:, ['token', 'n', 'k', 'weight']])
    not_observed['weight'] = not_observed['weight'].astype(int)
    not_observed = not_observed[not_observed['weight'] > 0]
    res = res.append(not_observed, ignore_index=True).sort_values(['token', 'n', 'k'])

    # index distinct (n, k) tuples
    nk = res[['n', 'k']].drop_duplicates().sort_values(['n', 'k'])
    nk['idx'] = np.arange(nk.shape[0])
    res = res.merge(nk, on=['n', 'k'])

    # convert to numpy
    n, k = nk['n'].values, nk['k'].values
    weights = np.zeros((n_tokens, nk.shape[0]))
    np.add.at(weights, (res['token'].values, res['idx'].values), res['weight'].values)

    return TokenStatistics(n, k, weights)


def get_observation_token_statistics(observations):
    """
    Compute token-level statistics from observations.

    Each observation is treated as its own entity, so that
    n_observations is always one.

    Parameters
    ----------
    observations : scipy.sparse.csr_matrix
        Binary observation matrix

    Returns
    -------
    token_stats : TokenStatistics
        Only the weights with n == 1 are populated.
    """
    n = np.array([1, 1])
    k = np.array([0, 1])
    weights = np.zeros((observations.shape[1], 2))
    weights[:, 1] = np.array(observations.sum(axis=0)).ravel()
    weights[:, 0] = observations.shape[0] - weights[:, 1]
    return TokenStatistics(n, k, weights)
