#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp

from bayax.utils.types import Callable, Optional


def christoffel_fk(
    g: Callable,
) -> Callable:
    r"""
    Calculate the christoffel symbols of the first kind (T_kij) given the metric

    Args:
        g: Metric

    Returns:
        Christoffel symbols first kind
    """

    def fn(x, v, u):
        term_v = jax.jvp(lambda x: g(x, u), (x,), (v,))[1]
        term_u = jax.jvp(lambda x: g(x, v), (x,), (u,))[1]
        grad_term = jax.grad(lambda x: jnp.dot(u, g(x, v)))(x)
        return 0.5 * (term_v + term_u - grad_term)

    # def fn(x, v):
    #     term = jax.jvp(lambda x: g(x, v), (x,), (v,))[1]
    #     grad_term = jax.grad(lambda x: jnp.dot(v, g(x, v)))(x)
    #     return term - 0.5 * grad_term

    return fn


def christoffel_sk(
    g: Callable,
    g_inv: Optional[Callable] = None,
    **kwargs
) -> Callable:
    r"""
    Calculate the christoffel symbols of the second kind (T^k_ij) given metric

    Args:
        g: Metric

    Returns:
        Christoffel symbols second kind
    """
    def fn(x, v, u):
        if g_inv is None:
            return jax.scipy.sparse.linalg.cg(lambda v: g(x, v), christoffel_fk(g)(x, v, u), **kwargs)[0]
        return g_inv(x, christoffel_fk(g)(x, v, u))

    return fn
