"""Utilities for Beta-Binomial likelihood."""

from autograd.scipy.special import gammaln

from .utils import damped_update


def beta_log_normalizer(alpha, beta):
    """Log-normalizer of Beta distribution."""
    return gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)


def beta_binomial_log_likelihood(alpha, beta, positive, total):
    """Log-likelihood of Beta-Binomial distribution."""
    return (
        beta_log_normalizer(alpha + positive, beta + total - positive) -
        beta_log_normalizer(alpha, beta))


class BetaParameters:
    """Container for parameters of Beta distribution."""
    def __init__(self, frequency=None, strength=None, alpha=None, beta=None):
        """
        Use either (frequency, strength) or (alpha, beta). If both are given,
        the first take precedence.
        """
        if frequency is not None and strength is not None:
            self.frequency = frequency
            self.strength = strength
        else:
            self.strength = alpha + beta
            self.frequency = alpha / self.strength

    # FIXME use more natural domain for interpolation
    def update(self, other, fraction=1.):
        """Update parameters."""
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
