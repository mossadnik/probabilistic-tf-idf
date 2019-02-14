import numpy as np


def sparse_iter_rows(mat):
    """iterate over csr_matrix rows (as slices)."""
    for row in range(mat.shape[0]):
        lo, hi = mat.indptr[row:row + 2]
        if hi > lo:
            yield row, slice(lo, hi)


def sparse_row_indices(mat):
    """get row indices for csr_matrix compatible with mat.indices."""
    res = np.zeros_like(mat.indices)
    for row, slc in sparse_iter_rows(mat):
        res[slc] = row
    return res
