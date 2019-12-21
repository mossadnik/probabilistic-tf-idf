# pylint: disable=missing-docstring
import numpy as np
from scipy.special import expit
from ptfidf.inference import NormalDist, BetaDist


def check_derivative(func, grad, x, eps=1e-7, atol=1e-7, rtol=1e-7):
    """Check derivative against finite difference."""
    return np.allclose(
        func(x + eps) - func(x),
        grad(x) * eps,
        atol=atol, rtol=rtol
    )


def test_normal_dist_grad():
    dist = NormalDist(0., 1.)
    x = np.linspace(-100, 100., 80)
    assert check_derivative(dist.lpdf, dist.lpdf_grad, x)


def test_beta_dist_grad():
    dist = BetaDist(.5, 1.7)
    x = expit(np.linspace(-8., 8., 80))
    assert check_derivative(dist.lpdf, dist.lpdf_grad, x)
