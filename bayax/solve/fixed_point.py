#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp

from typing import Callable, Optional


def fixed_point(
    update: Callable,
    initial: jnp.ndarray,
    tol: float = 1e-10,
    max_iter: int = 50,
    first: Optional[jnp.ndarray] = None,
):
    """
    Solve x = update(x) by fixed-point iteration

    Stops when ||x_new - x||_inf <= tol * (1 + ||x_new||_inf) or after
    max_iter updates. first optionally supplies update(initial) when the
    caller already evaluated it; it counts as the first iteration.

    Returns (x, converged).
    """
    def cond(carry):
        iteration, _, error, scale = carry
        return (iteration < max_iter) & (error > tol * scale)

    def step(carry):
        iteration, value, _, _ = carry
        candidate = update(value)
        error = jnp.linalg.norm(candidate - value, ord=jnp.inf)
        scale = 1.0 + jnp.linalg.norm(candidate, ord=jnp.inf)
        return iteration + 1, candidate, error, scale

    if first is None:
        iteration, value = jnp.array(0), initial
        error = jnp.array(jnp.inf, dtype=initial.dtype)
        scale = jnp.array(1.0, dtype=initial.dtype)
    else:
        iteration = jnp.asarray(max_iter > 0, dtype=jnp.int32)
        value = jnp.where(max_iter > 0, first, initial)
        error = jnp.where(
            max_iter > 0,
            jnp.linalg.norm(value - initial, ord=jnp.inf),
            jnp.inf,
        )
        scale = 1.0 + jnp.linalg.norm(value, ord=jnp.inf)

    _, value, error, scale = jax.lax.while_loop(
        cond,
        step,
        (iteration, value, error, scale),
    )
    return value, error <= tol * scale
