import numpy as np
from scipy.sparse import csr_matrix

from . import utils as ut


def get_log_proba(U, V, n_observations, token_frequencies, strength):
    r"""compute log observation probabilities.

    Parameters
    ----------
    U : csr_matrix
        binary document-term matrix for observations
    V : csr_matrix
        document-term count matrix for classes ($k_{zt}$ in maths part)
    n_observations : array
        number of observations for each class ($n_z$ in maths part)
    token_frequencies : array
        prior token probabilities ($\pi_t$ in maths part)
    strength : array
        prior strength parameter ($s_t$ in maths part)
    """
    max_observations = n_observations.max()
    n_tokens = U.shape[1]

    # compute p^0_t for all relevant n
    p0 = np.zeros((max_observations + 1, n_tokens), dtype=np.float32)
    n = np.arange(max_observations + 1, dtype=np.float32)[:, None]
    p0 = (strength * token_frequencies)[None, :] / (strength[None, :] + n)

    # k-independent terms
    unconstrained_term = np.log(1. - p0).sum(axis=1)
    t_in_x_term = U.dot(np.log(p0 / (1. - p0)).T)

    # k-dependent terms
    row = ut.sparse_row_indices(V)
    k = V.data  # count vectors
    n = n_observations[row]  # observation numbers
    t = V.indices  # token indices

    alpha = strength[t] * token_frequencies[t]
    beta = strength[t] * (1. - token_frequencies)[t]

    # interleave these two terms so that results can be shared
    t_in_k_cap_x_term = csr_matrix(
        (np.log((beta + n) / (beta + n - k)), V.indices, V.indptr),
        shape=V.shape
    )
    t_in_k_term = -np.array(t_in_k_cap_x_term.sum(axis=1)).ravel()  # note the minus
    t_in_k_cap_x_term.data += np.log((alpha + k) / alpha)

    # match names
    # -----------

    # \sum_{t \in x \cap k}
    log_proba = U.dot(t_in_k_cap_x_term.T)
    # \sum_{t \in k}
    log_proba.data += t_in_k_term[log_proba.indices]
    # \sum_{t \in x}
    row = ut.sparse_row_indices(log_proba)
    n = n_observations[log_proba.indices]
    log_proba.data += t_in_x_term[row, n]
    # \sum_t
    log_proba.data += unconstrained_term[n]
    
    return log_proba


def get_log_prior(X, token_frequencies):
    """compute log-prior of observations.
    
    Parameters
    ----------
    X : csr_matrix
        binary document-term matrix
    token_frequencies : array
        prior probabilities of terms
        
    Returns
    -------
    prior : array
        log-prior for each observation
    """
    prior = X.dot(np.log(token_frequencies / (1. - token_frequencies)))
    prior += np.log(1. - token_frequencies).sum()
    return prior


def get_proba(log_proba, log_prior, log_odds=0.):
    """compute normalized match probabilities.
    
    Takes into account possibility of creating a new partition,
    i.e. name not in list.

    Parameters
    ----------
    log_proba : csr_matrix
        unnormalized log-probabilities. Observations in rows,
        classes in columns.
    log_prior : array
        log-prior for each observation. Size must match number
        of rows of log_proba.
    log_odds : float, optional
        log-odds for having an existing class (vs creating a new one)
        
    Returns
    -------
    proba : csr_matrix
        normalized probabilities. Same shape and sparsity structure as
        log_proba. Sum over rows is the probability to belong to any
        existing class (Rows sum to less than one)
    """
    proba = csr_matrix(
        (np.empty_like(log_proba.data), log_proba.indices, log_proba.indptr),
        shape=log_proba.shape
    )

    for row, slc in ut.sparse_iter_rows(log_proba):
        lp = log_proba.data[slc] - log_prior[row] + log_odds
        lp_max = lp.max()
        lp -= lp_max
        norm = np.log(np.exp(lp).sum() + np.exp(-lp_max))
        proba.data[slc] = np.exp(lp - norm)
    
    return proba
