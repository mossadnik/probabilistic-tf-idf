"""Containers and aggregation functions for sufficient statistics."""

import numpy as np
from scipy import sparse
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
    def __init__(self, total_weights, positive_weights, negative_weights):
        self.total_weights = total_weights
        self.positive_weights = sparse.csr_matrix(positive_weights)
        self.negative_weights = sparse.csr_matrix(negative_weights)

    @property
    def weights_n(self):
        """Backward compatibility."""
        return self.total_weights[1:]

    @property
    def weights(self):
        """Backward compatibility."""
        return np.concatenate(
            [
                self.positive_weights.toarray()[:, None, 1:],
                self.negative_weights.toarray()[:, None, 1:]
            ],
            axis=1
        )

    @classmethod
    def from_entity_statistics(cls, entity_stats):
        """Aggregate EntityStatistics to token level."""
        _, n_tokens = entity_stats.counts.shape
        n_obs_vals, n_obs_cnts = np.unique(entity_stats.n_observations, return_counts=True)
        max_n_obs_count = n_obs_vals[-1]
        counts = entity_stats.counts.tocoo()

        # aggregate positive counts directly
        # cases with zero positive count are skipped
        # but can be recovered by summing over rows
        row = counts.col
        col = counts.data
        data = np.ones_like(col)
        positive_weights = sparse.coo_matrix(
            (data, (row, col)),
            shape=(n_tokens, max_n_obs_count + 1)
        )

        # aggregate negative counts indirectly
        # 1. Create default case from n_obs_vals, n_obs_counts
        row = [np.repeat(np.arange(n_tokens), n_obs_vals.size)]
        col = [np.tile(n_obs_vals, n_tokens)]
        data = [np.tile(n_obs_cnts, n_tokens)]

        # 2. Subtract cases that have any observations
        row.append(counts.col)
        col.append(entity_stats.n_observations[counts.row])
        data.append(np.full_like(counts.row, -1))

        # 3. Add cases with observations with correct counts
        row.append(counts.col)
        col.append(entity_stats.n_observations[counts.row] - counts.data)
        data.append(np.ones_like(counts.row))

        negative_weights = sparse.coo_matrix(
            (np.concatenate(data), (np.concatenate(row), np.concatenate(col))),
            shape=(n_tokens, max_n_obs_count + 1)
        )

        total_weights = np.zeros(n_obs_vals[-1] + 1, counts.dtype)
        total_weights[n_obs_vals] = n_obs_cnts
        # backward compatibility: convert to dense format
        return cls(total_weights, positive_weights, negative_weights)

    @classmethod
    def from_observations(cls, observations):
        """Aggregate observations to token level."""
        dtype = dict(dtype=np.int32)
        n_rows, n_tokens = observations.shape
        total_weights = np.array([0, n_rows], **dtype)


        counts = np.array(observations.sum(axis=0)).ravel()
        positive_weights = sparse.coo_matrix(
            (counts, (np.arange(n_tokens, **dtype), np.ones(n_tokens, **dtype))),
            shape=(n_tokens, 2)
        )
        negative_weights = sparse.coo_matrix(
            (n_rows - counts, (np.arange(n_tokens, **dtype), np.ones(n_tokens, **dtype))),
            shape=(n_tokens, 2)
        )

        return cls(total_weights, positive_weights, negative_weights)

    def add(self, other):
        """Add token counts."""
        dtype = dict(dtype=np.int32)
        if self.size != other.size:
            raise ValueError('Incompatible number of tokens: %d != %d' % (self.size, other.size))
        n_tokens = self.size
        n_max = 1 + max(self.max_count, other.max_count)
        total_weights = np.zeros(n_max, **dtype)
        positive_weights = sparse.coo_matrix((n_tokens, n_max), **dtype)
        negative_weights = sparse.coo_matrix((n_tokens, n_max), **dtype)
        for obj in [self, other]:
            total_weights[:obj.total_weights.size] += obj.total_weights
            padding = sparse.coo_matrix(
                (n_tokens, n_max - obj.max_count - 1),
                dtype=positive_weights.dtype
            )
            positive_weights += sparse.hstack([obj.positive_weights, padding])
            negative_weights += sparse.hstack([obj.negative_weights, padding])
        return self.__class__(total_weights, positive_weights, negative_weights)

    def copy(self):
        """Create new instance with copied data."""
        return self.__class__(
            self.total_weights.copy(),
            self.positive_weights.copy(),
            self.negative_weights.copy()
        )

    def get_unique_weights(self):
        """Return deduplicated weights.

        Returns
        -------
        (positive_weights, negative_weights, index, inverse, counts) : tuple
            positive_weights and negative_weights are deduplicated.
            index, inverse, counts are defined like in numpy.unique
        """
        n = self.total_weights.size
        # Need to find simultaneous unique rows of positive_weights and negative weights
        projection = np.random.uniform(-1., 1., size=(n, min(n, 10)))
        hashed = np.c_[self.positive_weights.dot(projection), self.negative_weights.dot(projection)]
        _, index, inverse, counts = np.unique(
            hashed, axis=0,
            return_counts=True, return_index=True, return_inverse=True
        )
        return self.positive_weights[index], self.negative_weights[index], index, inverse, counts

    def __repr__(self):
        return 'TokenStatistics(%d tokens)' % self.size

    @property
    def size(self):
        """Get number of tokens."""
        return self.positive_weights.shape[0]

    @property
    def max_count(self):
        """Get max number of entity observations."""
        return self.total_weights.size - 1
