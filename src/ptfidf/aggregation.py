"""Containers and aggregation functions for sufficient statistics."""

import numpy as np
from scipy.sparse import csr_matrix

from . import utils as ut


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

    def get_token_statistics(self):
        """Aggregate sufficient statistics to token level."""
        # group by (token, (n, k)) and count
        rows = ut.sparse_row_indices(self.counts)
        groups, cnt = np.unique(
            np.r_[self.counts.indices[None, :], _nk2idx(self.n_observations[rows], self.counts.data)[None, :]],
            return_counts=True,
            axis=1)

        # compute relevant (n, k) indices
        index = np.unique(groups[1])  # from data
        n, k = _idx2nk(index)
        n_unique = np.unique(n)
        zero_index = _nk2idx(n_unique, np.zeros_like(n_unique))  # (n, k=0) for all n in data
        full_index = np.union1d(index, zero_index)  # all
        n, k = _idx2nk(full_index)

        # observations with k > 0 from data
        weights = np.zeros((self.counts.shape[1], full_index.size), dtype=np.int64)
        idx = np.searchsorted(full_index, np.arange(full_index.max() + 1))
        np.add.at(weights, (groups[0], idx[groups[1]]), cnt)
        # add observations for k = 0
        _, n_observations_counts = np.unique(self.n_observations, return_counts=True)
        bnd = bnd = np.concatenate([np.searchsorted(full_index, zero_index), [full_index.size]])
        for i in range(bnd.size - 1):
            weights[:, bnd[i]] = n_observations_counts[i] - weights[:, bnd[i]:bnd[i + 1]].sum(axis=1)
        return TokenStatistics(n, k, weights)


def get_entity_statistics(observations, assignments, n_classes=None):
    """Aggregate observations to entity level.

    Parameters
    ----------
    observations : scipy.sparse.csr_matrix
        binary document-term matrix
    assignment : numpy.ndarray
        class labels. Must have y.size == X.shape[0]
    n_classes : int, optional
        number of classes (defaults to `y.max() + 1`)

    Returns
    -------
    EntityStatistics
    """
    if assignments.size != observations.shape[0]:
        raise ValueError('Incompatible shapes.')
    n = assignments.size
    if n_classes is None:
        n_classes = assignments.max() + 1
    dt = np.int32
    data, row, col = np.ones(n, dtype=dt), assignments.astype(dt), np.arange(n, dtype=dt)
    assignments_mat = csr_matrix((data, (row, col)), shape=(n_classes, observations.shape[0]))
    counts = assignments_mat.dot(observations)
    n_observations = np.array(assignments_mat.sum(axis=1)).ravel()

    return EntityStatistics(counts, n_observations)


def _nk2idx(n, k):
    """Convert n, k to integer index."""
    return ((n - 1) * (n + 2)) // 2 + k


def _idx2nk(idx):
    """Convert integer index to n, k.

    inverse of _nk2idx.
    """
    n = np.floor(-.5 + .5 * np.sqrt(9 + 8 * idx)).astype(int)
    k = idx - (n - 1) * (n + 2) // 2
    return n, k


class TokenStatistics(object):
    """Container for token-level sufficient statistics."""
    def __init__(self, n, k, weights):
        self.n = n
        self.k = k
        self.weights = weights.astype(np.float32)

    @property
    def size(self):
        """get number of tokens."""
        return self.weights.shape[0]

    def __repr__(self):
        s = 'TokenStatistics({} tokens)'
        return s.format(self.size)


    def add(self, other):
        """Merge counts, add weights."""
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
    TokenStatistics
        Only the weights with n == 1 are populated.
    """
    n = np.array([1, 1])
    k = np.array([0, 1])
    weights = np.zeros((observations.shape[1], 2))
    weights[:, 1] = np.array(observations.sum(axis=0)).ravel()
    weights[:, 0] = observations.shape[0] - weights[:, 1]
    return TokenStatistics(n, k, weights)
