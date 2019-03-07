"""Low-level observation model functions."""

import numpy as np
from scipy.sparse import csr_matrix

from . import utils as ut


def get_log_proba(observations, entity_stats, token_frequencies, strength):
    r"""Compute log observation probabilities.

    Parameters
    ----------
    observations : scipy.sparse.csr_matrix
        binary document-term matrix for observations
    entity_stats : EntityStats
        Entity-level word count statistics
    token_frequencies : numpy.ndarray
        prior token probabilities ($\pi_t$ in maths part)
    strength : numpy.ndarray
        prior strength parameter ($s_t$ in maths part)
    """
    counts = entity_stats.counts
    n_tokens = observations.shape[1]
    n_obs = entity_stats.n_observations
    max_observations = n_obs.max()

    # compute p^0_t for all relevant n
    p0 = np.zeros((max_observations + 1, n_tokens), dtype=np.float32)
    n = np.arange(max_observations + 1, dtype=np.float32)[:, None]
    p0 = (strength * token_frequencies)[None, :] / (strength[None, :] + n)

    # k-independent terms
    unconstrained_term = np.log(1. - p0).sum(axis=1)

    # k-dependent terms
    k = counts.data  # count vectors
    n = n_obs[ut.sparse_row_indices(counts)]  # observation numbers
    t = counts.indices  # token indices

    alpha = strength[t] * token_frequencies[t]
    beta = strength[t] * (1. - token_frequencies)[t]

    # interleave these two terms so that results can be shared
    t_in_k_cap_x_term = csr_matrix(
        (np.log((beta + n) / (beta + n - k)), counts.indices, counts.indptr),
        shape=counts.shape
    )
    t_in_k_term = -np.array(t_in_k_cap_x_term.sum(axis=1)).ravel()  # note the minus
    t_in_k_cap_x_term.data += np.log((alpha + k) / alpha)

    # everything above can be precomputed
    # \sum_{t \in observations \cap k}
    log_proba = observations.dot(t_in_k_cap_x_term.T)
    # \sum_{t \in k}
    log_proba.data += t_in_k_term[log_proba.indices]
    # \sum_{t \in x}
    n = n_obs[log_proba.indices]
    t_in_x_term = observations.dot(np.log(p0 / (1. - p0)).T)
    row = ut.sparse_row_indices(log_proba)
    log_proba.data += t_in_x_term[row, n]
    # \sum_t
    log_proba.data += unconstrained_term[n]

    return log_proba


def get_log_prior(observations, token_frequencies):
    """Compute log-prior of observations.

    Parameters
    ----------
    observations : (N, T) scipy.sparse.csr_matrix
        binary document-term matrix
    token_frequencies : (T,) numpy.ndarray
        prior probabilities of terms

    Returns
    -------
    prior : numpy.ndarray
        log-prior for each observation
    """
    prior = observations.dot(np.log(token_frequencies / (1. - token_frequencies)))
    prior += np.log(1. - token_frequencies).sum()
    return prior


def get_proba(log_proba, log_prior, log_odds=0.):
    """Compute normalized match probabilities.

    Takes into account possibility of creating a new partition,
    i.e. name not in list.

    Parameters
    ----------
    log_proba : scipy.sparse.csr_matrix
        unnormalized log-probabilities. Observations in rows,
        classes in columns.
    log_prior : numpy.ndarray
        log-prior for each observation. Size must match number
        of rows of log_proba.
    log_odds : float, optional
        log-odds for assignment to an existing class vs creating a new one.

    Returns
    -------
    proba : scipy.sparse.csr_matrix
        normalized probabilities. Same shape and sparsity structure as
        log_proba. Sum over rows is the probability to belong to any
        existing class (Rows sum to less than one).
    """
    proba = csr_matrix(
        (np.empty_like(log_proba.data), log_proba.indices, log_proba.indptr),
        shape=log_proba.shape
    )

    for row, slc in ut.sparse_iter_rows(log_proba):
        lp = log_proba.data[slc] - log_prior[row] + log_odds
        lp_max = lp.max()
        lp -= lp_max
        norm = np.log(np.sum(np.exp(lp)) + np.exp(-lp_max))
        proba.data[slc] = np.exp(lp - norm)

    return proba


def sample_assignments(proba):
    """Randomly assing observations according to distribution.

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
    for row, slc in ut.sparse_iter_rows(proba):
        choice = np.searchsorted(np.cumsum(proba.data[slc]), rnd[row])
        if choice < slc.stop - slc.start:
            assignments[row] = proba.indices[slc][choice]
    return assignments
