#!/usr/bin/env python
# encoding: utf-8

import jax.numpy as jnp
import jax.random as jr
from bayax.utils.types import Callable


def rff(
    dim: int,
    m: int = 512,
    seed: int = 0,
) -> Callable:
    r"""
    Random Fourier features of the squared exponential kernel.

    Returns phi(x, l, tau), mapping x of shape (..., dim) to (..., m) features with
    phi(x, l, tau) @ phi(y, l, tau) ≈ rbf(x, y, l, tau) for scalar l and
    ≈ ard(x, y, l, tau) for l of shape (dim,). The frequencies are drawn once here
    and scaled by the lengthscales at evaluation, so phi is differentiable in l.

    Args:
        dim: Input dimension
        m: Number of features
        seed: Seed of the frequencies and phases
    """
    k1, k2 = jr.split(jr.key(seed))
    z = jr.normal(k1, shape=(m, dim))
    b = jr.uniform(k2, shape=(m,), maxval=2 * jnp.pi)

    def phi(x, l, tau=1.0):
        return jnp.exp(tau / 2) * jnp.sqrt(2 / m) * jnp.cos(x @ (z / jnp.exp(l)).T + b)
    return phi
