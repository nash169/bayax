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
    Return the Hamiltonian using geom(q) -> bayax.geom.embedding.EmbeddingGeom

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
    Return the position gradient using geom(q) -> bayax.geom.embedding.EmbeddingGeom

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


def d2Hdqdp_geom(
    geom: Callable,
) -> Callable:
    """
    Return the momentum derivative A = d(dH/dq)/dp of dHdq_geom

    The potential and log determinant gradients do not depend on p. With v = G^-1 p,
    the rest of dH/dq is -cfk_mv(v, "ki") = p^T dG^-1 p / 2, so
    A dp = -grad_q(v^T G w) = -(cfk_mv(v, w, "ki") + cfk_mv(w, v, "ki")) with w = G^-1 dp,
    A^T z = (DG^-1[z]) p = -G^-1 (DG[z]) v.
    pdot = -dH/dq has momentum derivative -A. Returns ((jvp, vjp), cache) for vectors,
    reusing g at cache[0]; without a supplied cache, return the geom-only cache (g,).
    """
    def fn(t, x, u=None, cache=None):
        q, p = jnp.split(x, 2)
        if cache is None:
            g = geom(q)
            cache = (g,)
        else:
            g = cache[0]
        v = g.metric_solve(p)

        def jvp(dp):
            w = g.metric_solve(dp)
            return -(g.cfk_mv(v, w, contraction="ki") + g.cfk_mv(w, v, contraction="ki"))

        def vjp(z):
            return -g.metric_solve(g.metric_grad_mv(z, v))

        return (jvp, vjp), cache

    return fn
