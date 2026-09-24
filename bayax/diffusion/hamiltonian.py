#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
from numpy import half

from bayax.utils.types import Callable, Optional
from bayax.operators.linear_operator import LinearOperator
from bayax.operators import DenseOperator, LowRankOperator, PSDOperator
from bayax.geom.embedding import pullmetric, christoffel_fk


def H_geom(
    energy: Callable,
    geom: Callable,
    logdet: bool = False
) -> Callable:
    """
    Return the Hamiltonian using geom(q) -> EmbeddingCache

    The optional cache is (g, potential_energy), where potential_energy
    includes the half log determinant and Gaussian normalization at q.
    Returns (value, cache).
    """
    def fn(t, x, u=None, cache=None):
        q, p = jnp.split(x, 2)
        if cache is None:
            g = geom(q)
            potential_energy = energy(q) + 0.5 * (q.shape[0] * jnp.log(2 * jnp.pi))
            if logdet:
                potential_energy += 0.5 * g.metric_logdet()
            cache = (g, potential_energy)
        else:
            g, potential_energy = cache

        return potential_energy + 0.5 * (p @ g.metric_solve(p)), cache

    return fn


def dHdp_geom(
    geom: Callable,
) -> Callable:
    """
    Return (momentum gradient, cache), reusing g at cache[0]

    Without a supplied cache, return the geom-only cache (g,).
    """
    def fn(t, x, u=None, cache=None):
        q, p = jnp.split(x, 2)
        if cache is None:
            g = geom(q)
            cache = (g,)
        else:
            g = cache[0]
        return g.metric_solve(p), cache

    return fn


def dHdq_geom(
    grad_energy: Callable,
    geom: Callable,
    logdet: bool = False
) -> Callable:
    """
    Return the position gradient using geom(q) -> EmbeddingCache

    The optional cache is (g, potential_grad), where potential_grad is
    grad_energy(q) plus the half log-determinant gradient at q.
    Returns (position gradient, cache).
    """
    def fn(t, x, u=None, cache=None):
        q, p = jnp.split(x, 2)
        if cache is None:
            g = geom(q)
            potential_grad = grad_energy(q)
            if logdet:
                potential_grad += g.half_grad_logdet()
            cache = (g, potential_grad)
        else:
            g, potential_grad = cache

        v = g.metric_solve(p)
        return potential_grad - g.cfk_mv(v, contraction="ki"), cache

    return fn


# def dHdq_geom_bkp(
#     grad_energy: Callable,
#     metric: Callable,
#     christoffels: Callable,
# ) -> Callable:
#     r"""
#     Calculate the Hamiltonian gradient with respect to q at fixed momentum
#
#     The log-determinant gradient contracts over the columns of the array
#     B = g.inv_sqrt(), where B @ B.T is the inverse metric.
#
#     Args:
#         grad_energy: Energy gradient
#         metric: Callable returning a metric operator with solve and inv_sqrt
#         christoffels: Contracted first-kind Christoffels (q, w, v)
#     """
#     def fn(t, x, u=None, cache=None):
#         q, p = jnp.split(x, 2)
#
#         if cache is None:
#             g = metric(q)
#             g_inv_sqrt = g.metric_inv.sqrtf()._mat
#
#             def contractions(w):
#                 return jax.vmap(
#                     lambda b: christoffels(x, w, b),
#                     in_axes=1,
#                     out_axes=1,
#                 )(g_inv_sqrt)
#
#             potential_grad = jax.linear_transpose(contractions, x)(g_inv_sqrt)[0] + grad_energy(q)
#         else:  # position dependent quantities
#             g, potential_grad = cache
#
#         # kinetic grad (q and p dependency cannot be cached)
#         v = g.metric_solve(p)
#         kinetic_grad = -jax.linear_transpose(lambda w: christoffels(q, w, v), q)(v)[0]
#
#         return potential_grad + kinetic_grad, cache
#
#     return fn
