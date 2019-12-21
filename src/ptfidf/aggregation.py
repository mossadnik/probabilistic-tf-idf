"""Containers and aggregation functions for sufficient statistics."""

import numpy as np
from scipy.sparse import csr_matrix, hstack


class EntityStatistics:
    """Entity-level sufficient statistics."""
    def __init__(self, counts, n_observations):
        self.counts = counts
        self.n_observations = n_observations

    @classmethod
    def from_observations(cls, observations, labels, n_classes=None):
        """Aggregate observations to entity level.

        Parameters
        ----------
        observations : scipy.sparse.csr_matrix
            binary document-term matrix
        labels : numpy.ndarray
            class labels. Must have y.size == X.shape[0]
        n_classes : int, optional
            number of classes (defaults to `y.max() + 1`)

        Returns
        -------
        EntityStatistics
        """
        if labels.size != observations.shape[0]:
            raise ValueError('Incompatible shapes.')
        n = labels.size
        if n_classes is None:
            n_classes = labels.max() + 1
        dtype = np.int32
        data, row, col = np.ones(n, dtype=dtype), labels.astype(dtype), np.arange(n, dtype=dtype)
        assignments_mat = csr_matrix((data, (row, col)), shape=(n_classes, observations.shape[0]))
        counts = assignments_mat.dot(observations)
        n_observations = np.array(assignments_mat.sum(axis=1)).ravel()

        return cls(counts, n_observations)

    @property
    def n_tokens(self):
        """Get number of distinct tokens."""
        return self.counts.shape[1]

    @property
    def size(self):
        """Get number of entities."""
        return self.n_observations.size

    def __repr__(self):
        s = 'EntityStatistics({} entities, {} observations)'
        return s.format(self.counts.shape[0], self.n_observations.sum())

    def extend_tokens(self, n_tokens):
        """Increase the number of tokens.

        New tokens are appended at the end with all counts
        set to zero.

        Parameters
        ----------
        n_tokens : int >= 0
            Number of tokens to add.
        """
        if n_tokens == 0:
            return self
        new_block = csr_matrix((self.counts.shape[0], n_tokens), dtype=self.counts.dtype)
        self.counts = hstack([self.counts, new_block])
        return self

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


class TokenStatistics:
    """Container for count aggregation to token level.

    The constructor is not meant to be used directly.
    New instances are created using the class methods

    `TokenStatistics.from_entity_statistics`
    `TokenStatistics.from_observations`
    """
    def __init__(self, weights_n, weights):
        self.weights_n = weights_n
        self.weights = weights

    @classmethod
    def from_entity_statistics(cls, entity_stats):
        """Aggregate EntityStatistics to token level."""
        vals, cnts = np.unique(entity_stats.n_observations, return_counts=True)
        n_max = vals[-1]
        n_entities, n_tokens = entity_stats.counts.shape

        weights_n = np.zeros(n_max + 1, dtype=np.int32)
        weights_n[vals] = cnts

        counts = entity_stats.counts.tocoo()

        weights = np.zeros((n_tokens, 2, n_max + 1), dtype=np.int32)
        # positive cases
        np.add.at(weights, (counts.col, 0, counts.data), 1)
        weights[:, 0, 0] = n_entities - weights[:, 0].sum(axis=-1)

        # negative cases, copy over n-weights as default
        weights[:, 1] = weights_n[None, :]
        # add weights for observed values
        np.add.at(
            weights,
            (counts.col, 1, entity_stats.n_observations[counts.row] - counts.data),
            1
        )
        # remove default weights for observed values
        np.add.at(weights, (counts.col, 1, entity_stats.n_observations[counts.row]), -1)
        return cls(weights_n[1:], weights[:, :, 1:])

    @classmethod
    def from_observations(cls, observations):
        """Aggregate observations to token level."""
        n_rows, n_cols = observations.shape
        weights_n = np.array([n_rows], dtype=np.int32)

        weights = np.zeros((n_cols, 2, 1), dtype=np.int32)
        counts = np.array(observations.sum(axis=0)).ravel()
        weights[:, 0, 0] = counts
        weights[:, 1, 0] = n_rows - counts
        return cls(weights_n, weights)

    def add(self, other):
        """Add token counts."""
        if self.size != other.size:
            raise ValueError('Incompatible number of tokens: %d != %d' % (self.size, other.size))
        n_tokens = self.size
        n_max = max(self.max_count, other.max_count)
        weights_n = np.zeros(n_max, dtype=np.int32)
        weights = np.zeros((n_tokens, 2, n_max))
        for obj in [self, other]:
            weights_n[:obj.weights_n.size] += obj.weights_n
            weights[:, :, :obj.weights.shape[-1]] += obj.weights
        return self.__class__(weights_n, weights)

    def copy(self):
        """Create new instance with copied data."""
        return self.__class__(self.weights_n.copy(), self.weights.copy())

    def __repr__(self):
        return 'TokenStatistics(%d tokens)' % self.size

    @property
    def size(self):
        """Get number of tokens."""
        return self.weights.shape[0]

    @property
    def max_count(self):
        """Get max number of entity observations."""
        return self.weights.shape[-1]
