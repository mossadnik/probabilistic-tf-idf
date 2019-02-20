import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

from .. import utils as ut


def get_group_statistics(X, y):
    """Aggregate token sufficient statistics to group level.
    
    Parameters
    ----------
    X : csr_matrix
        binary document-term matrix
    y : array
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


# TODO tokens that never occur are dropped. Should they be included?
# TODD integrate with group level aggregation
def get_token_statistics(X, labels):
    """compute token statistics for all tokens from labeled name list.
    
    Parameters
    ----------
    X : csr_matrix
        token counts for each name
    labels : pd.Series
        entity associated to each name. Must have
        labels.size == X.shape[0]
    
    Returns
    -------
    pd.DataFrame
        columns are
        token   token id
        n       number of names in entity
        k       number of occurences of token
        cnt     multiplicity of (token, n, k) 
        
        primary key is (token, n, k)
    """
    tokens = (
        ut.sparse_to_frame(X)
        .rename(columns={'col': 'token', 'data': 'k'})
    )

    n_observations = pd.Series(labels.value_counts(), name='n')
    nobs_counts = (
        pd.Series(n_observations.value_counts(), name='total')
        .reset_index()
        .rename(columns={'index': 'n'}))

    # group token observations by entity
    res = (
        tokens
        .assign(entity=tokens['row'].map(labels))
        .groupby(['entity', 'token'])[['k']]
        .sum()
        .reset_index())

    # add in number of observations and group by token
    res['n'] = res['entity'].map(n_observations)
    res = (
        res.groupby(['token', 'k', 'n'])
        .size()
        .reset_index()
        .rename(columns={0: 'weight'}))

    # add counts of entities without observation of token
    not_observed = (
        ut.cartesian_product(
            res[['token']].drop_duplicates(), nobs_counts)
        .merge(
            res.groupby(['token', 'n'])[['weight']].sum(),
            left_on=['token', 'n'],
            right_index=True,
            how='left')
        .fillna(0))
    not_observed['weight'] = (not_observed['total'] - not_observed['weight']).astype(int)
    not_observed['k'] = 0
    not_observed.drop('total', axis=1, inplace=True)
    res = res.append(not_observed, sort=False).sort_values(['token', 'n', 'k'])

    return res[res['weight'] > 0]
