from autograd.scipy.special import gammaln


def beta_log_normalizer(a, b):
    """Log-normalizer of Beta distribution."""
    return gammaln(a) + gammaln(b) - gammaln(a + b)


def beta_binomial_log_likelihood(a, b, k, n):
    """Log-likelihood of Beta-Binomial distribution."""
    return beta_log_normalizer(a + k, b + n - k) - beta_log_normalizer(a, b)
