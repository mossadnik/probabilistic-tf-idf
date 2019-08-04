"""Utility functions."""

import numpy as np
from scipy import sparse


def sparse_iter_rows(mat):
    """Iterate over csr_matrix rows (as slices)."""
    for row in range(mat.shape[0]):
        start, stop = mat.indptr[row:row + 2]
        if stop > start:
            yield row, slice(start, stop)


def sparse_row_indices(mat):
    """Get row indices for csr_matrix in array like mat.indices."""
    res = np.array(np.zeros_like(mat.indices))
    for row, slc in sparse_iter_rows(mat):
        res[slc] = row
    return res


def update(old, new, fraction=1.):
    """inplace damped update."""
    old *= (1. - fraction)
    old += fraction * new


def get_normalized_proba(log_proba, log_prior=None):
    """Compute normalized probabilities from sparse log-probabilities.

    Parameters
    ----------
    log_proba : scipy.sparse.csr_matrix
        Observations in rows, classes in columns.

    log_prior : numpy.ndarray
        log-prior for each row to belong to a new class.
        size has to match number of rows of log_proba.
        If set to None (default), computes normalized classification
        probabilities.
        Otherwise, computes classification probabilities that sum to
        less than unity. The remaining weight is the probability to
        belong to a new class.

    Returns
    -------
    proba : scipy.sparse.csr_matrix
    """
    rowmax = log_proba.max(axis=1).toarray().ravel()
    res = log_proba.tocoo()
    res.data -= rowmax[res.row]
    res.data = np.exp(res.data)
    res = sparse.csr_matrix(res)
    norm = res.sum(axis=1).A.ravel()
    if log_prior is not None:
        norm += np.exp(-rowmax + log_prior)
    return sparse.csr_matrix(res.multiply(1. / np.maximum(1e-12, norm[:, None])))


def sample_assignments(proba):
    """Randomly assign observations according to distribution.

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
    for row, slc in sparse_iter_rows(proba):
        choice = np.searchsorted(np.cumsum(proba.data[slc]), rnd[row])
        if choice < slc.stop - slc.start:
            assignments[row] = proba.indices[slc][choice]
    return assignments
