import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

from .. import utils as ut


def get_group_statistics(X, y):
    """
    Aggregate token sufficient statistics to group level.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        binary document-term matrix
    y : numpy.ndarray
        group labels. Must have y.size == X.shape[0]

    Returns
    -------
    counts : csr_matrix
        token count vectors on group level. Shape
        (max(y) + 1, X.shape[1]).
    n_observations : array
        number of observations for each group. Shape
        n_observations.size == counts.shape[0].
    """
    if y.size != X.shape[0]:
        raise ValueError('Incompatible shapes.')
    n = y.size
    dt = np.int32
    data, row, col = np.ones(n, dtype=dt), y.astype(dt), np.arange(n, dtype=dt)
    association = csr_matrix((data, (row, col)))
    counts = association.dot(X)
    n_observations = np.array(association.sum(axis=1)).ravel()

    return counts, n_observations


def compress_group_statistics(counts, n_observations):
    """
    Aggregate group-level sufficient statistics into distinct likelihood terms.

    Parameters
    ----------
    counts : scipy.sparse.csr_matrix
        token counts per group.
    n_observations : numpy.ndarray
        number of observations per group.

    Returns
    -------
    pandas.DataFrame
        Sufficient statistics and weights for all token
        likelihood terms. Columns ['token', 'n', 'k', 'weight']
        with primary key ['token', 'n', 'k'].
    """
    n_tokens = counts.shape[1]
    distinct_nobs, count_nobs = np.unique(n_observations, return_counts=True)

    # Aggregate counts into distinct likelihood terms with weights / multiplicities
    res = ut.sparse_to_frame(counts).rename(columns={'row': 'group', 'col': 'token', 'data': 'k'})
    res['n'] = n_observations[res['group'].values]
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

    return n, k, weights
