#!/usr/bin/env python
# encoding: utf-8

import jax

from bayax.utils.types import Callable, Optional
from bayax.operators import DenseOperator, PSDOperator


def brownian(
        cov: PSDOperator,
) -> Callable:
    def fn(t, x, u):
        return None, cov

    return fn


def brownian_geom(
        geom: Callable,
        include_christoffels: bool = True,
) -> Callable:
    """Return Brownian dynamics using the geometry cache returned by geom(x).

    Set include_christoffels=False to omit the Christoffel drift correction.
    """
    def fn(t, x, u):
        g = geom(x)
        if include_christoffels:
            factor = g.metric_inv.sqrtf()
            B = factor._mat if isinstance(factor, DenseOperator) else factor.dense()._mat
            drift = -0.5 * g.metric_solve(g.cfk_mv(B, contraction="ij"))
        else:
            drift = None

        return drift, g.metric_inv

    return fn


def brownian_geometric(
        metric: Callable,
        christoffels: Optional[Callable] = None,
) -> Callable:
    def fn(t, x, u):
        g = metric(x)

        if christoffels is not None:
            drift = -0.5 * g.inv(jax.vmap(lambda u: christoffels(x, u, u), in_axes=(1,))(g.inv_sqrt).sum(axis=0))
        else:
            drift = None

        return drift, g.inv

    return fn
