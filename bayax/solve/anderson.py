#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp

from typing import Callable


def anderson(
    update: Callable,
    initial: jnp.ndarray,
    tol: float = 1e-10,
    max_iter: int = 50,
    memory: int = 5,
    beta: float = 1.0,
    reg: float = 1e-12,
):
    """
    Solve x = update(x) by Anderson acceleration of the fixed-point iteration

    Keeps the last `memory` differences of the iterates g_i = update(x_i) and
    residuals f_i = g_i - x_i, and takes x_new = x + beta f - (dX + beta dF) gamma
    with gamma = argmin ||f - dF gamma||, solved from the memory x memory
    normal equations with a relative ridge reg. Only update evaluations are
    needed (no Jacobian products); it converges for mildly divergent or slowly
    contracting fixed points. memory = 0 is plain (relaxed) fixed-point iteration.

    Stops when ||update(x) - x||_inf <= tol * (1 + ||update(x)||_inf) or after
    max_iter updates, as fixed_point. Returns (update(x), converged).
    """
    n, m = initial.size, memory

    def cond(carry):
        iteration, _, _, _, _, _, _, error, scale = carry
        return (iteration < max_iter) & (error > tol * scale)

    def step(carry):
        iteration, x, x_prev, f_prev, dX, dF, g_last, _, _ = carry
        g = update(x)
        f = g - x
        error = jnp.linalg.norm(f, ord=jnp.inf)
        scale = 1.0 + jnp.linalg.norm(g, ord=jnp.inf)
        if m == 0:
            return iteration + 1, x + beta * f, x, f, dX, dF, g, error, scale
        # store the newest differences in a circular buffer (none at the first iteration)
        slot = (iteration - 1) % m
        store = iteration > 0
        dX = jnp.where(store, dX.at[slot].set(x - x_prev), dX)
        dF = jnp.where(store, dF.at[slot].set(f - f_prev), dF)
        valid = jnp.arange(m) < jnp.minimum(iteration, m)
        A = dF @ dF.T
        ridge = reg * (jnp.trace(A) + 1e-300)
        A = jnp.where(valid[:, None] & valid[None, :], A, 0.0) + jnp.diag(jnp.where(valid, ridge, 1.0))
        gamma = jnp.where(valid, jnp.linalg.solve(A, jnp.where(valid, dF @ f, 0.0)), 0.0)
        x_new = x + beta * f - (dX + beta * dF).T @ gamma
        return iteration + 1, x_new, x, f, dX, dF, g, error, scale

    zeros = jnp.zeros((max(m, 1), n), dtype=initial.dtype)
    _, _, _, _, _, _, g, error, scale = jax.lax.while_loop(
        cond,
        step,
        (
            jnp.array(0), initial, initial, jnp.zeros_like(initial), zeros, zeros, initial,
            jnp.array(jnp.inf, dtype=initial.dtype), jnp.array(1.0, dtype=initial.dtype),
        ),
    )
    return g, error <= tol * scale
