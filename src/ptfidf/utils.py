import numpy as np
import pandas as pd


def sparse_iter_rows(mat):
    """Iterate over csr_matrix rows (as slices)."""
    for row in range(mat.shape[0]):
        lo, hi = mat.indptr[row:row + 2]
        if hi > lo:
            yield row, slice(lo, hi)


def sparse_row_indices(mat):
    """Get row indices for csr_matrix in array like mat.indices."""
    res = np.zeros_like(mat.indices)
    for row, slc in sparse_iter_rows(mat):
        res[slc] = row
    return res


def sparse_to_frame(mat):
    """Convert sparse matrix into long-format DataFrame."""
    return pd.DataFrame({'row': sparse_row_indices(mat), 'col': mat.indices, 'data': mat.data})
