#!/usr/bin/env python
# encoding: utf-8

import jax

from bayax.utils.types import Callable, Optional
from bayax.operators import DenseOperator, PSDOperator


def langevin(
        grad_energy: Callable,
        cov: PSDOperator,
) -> Callable:
    def fn(t, x, u):
        return -0.5 * grad_energy(x), cov

    return fn


def langevin_geom(
        grad_energy: Callable,
        geom: Callable,
        include_christoffels: bool = True,
) -> Callable:
    """Return Langevin dynamics using the geometry cache returned by geom(x).

    Set include_christoffels=False to omit the Christoffel drift correction.
    """
    def fn(t, x, u):
        g = geom(x)
        force = grad_energy(x)
        if include_christoffels:
            factor = g.metric_inv.sqrtf()
            B = factor._mat if isinstance(factor, DenseOperator) else factor.dense()._mat
            force = force + g.cfk_mv(B, contraction="ij")
        drift = -0.5 * g.metric_solve(force)

        return drift, g.metric_inv

    return fn


def langevin_geometric(
        grad_energy: Callable,
        metric: Callable,
        christoffels: Optional[Callable] = None,
) -> Callable:
    def fn(t, x, u):
        g = metric(x)

        if christoffels is not None:
            drift = -0.5 * g.inv(grad_energy(x) + jax.vmap(lambda u: christoffels(x, u, u), in_axes=(1,))(g.inv_sqrt).sum(axis=0))
        else:
            drift = -0.5 * g.inv(grad_energy(x))

        return drift, g.inv

    return fn
