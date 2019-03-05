from autograd.scipy.special import gammaln

from ..utils import damped_update


def beta_log_normalizer(a, b):
    """Log-normalizer of Beta distribution."""
    return gammaln(a) + gammaln(b) - gammaln(a + b)


def beta_binomial_log_likelihood(a, b, k, n):
    """Log-likelihood of Beta-Binomial distribution."""
    return beta_log_normalizer(a + k, b + n - k) - beta_log_normalizer(a, b)


class BetaParameters(object):
    """Container for parameters of Beta distribution."""
    def __init__(self, alpha=None, beta=None, frequency=None, strength=None):
        """
        Use either (alpha, beta) or (frequency, strength). If both are given,
        alpha, beta takes precedence.
        """
        if alpha is not None and beta is not None:
            strength = alpha + beta
            frequency = alpha / strength
        self.frequency = frequency
        self.strength = strength

    def update(self, other, fraction=1.):
        damped_update(self.frequency, other.frequency, fraction)
        damped_update(self.strength, other.strength, fraction)

    @property
    def alpha(self):
        """Convert to standard parametrization."""
        return self.frequency * self.strength

    @property
    def beta(self):
        """Convert to standard parametrization."""
        return (1. - self.frequency) * self.strength
