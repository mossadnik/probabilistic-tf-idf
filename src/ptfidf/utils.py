"""Utility functions."""

import numpy as np


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
