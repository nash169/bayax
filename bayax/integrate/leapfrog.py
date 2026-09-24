#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp


def leapfrog(f, t, x, u, dt):
    """Advance separable dynamics ``f=(pdot, qdot)`` by one leapfrog step."""
    pdot, qdot = f
    q, p = jnp.split(x, 2)

    p_half = p + 0.5 * dt * pdot(t, x, u)

    q_new = q + dt * qdot(t, jnp.concatenate((q, p_half)), u)

    p_new = p_half + 0.5 * dt * pdot(
        t + dt,
        jnp.concatenate((q_new, p_half)),
        u,
    )

    return jnp.concatenate((q_new, p_new))


def generalized_leapfrog(f, t, x, u, dt, tol=1e-10, max_iter=50):
    """
    Advance nonseparable dynamics ``f=(pdot, qdot)`` implicitly

    Callbacks may return a derivative or (derivative, cache). A returned
    pdot cache must also be accepted by qdot via its cache keyword. Reuse
    this cache only at the same position and time within the step.
    """
    pdot, qdot = f
    q, p = jnp.split(x, 2)

    def evaluate(fn, time, state, cache=None):
        result = fn(time, state, u) if cache is None else fn(time, state, u, cache=cache)
        return result if isinstance(result, tuple) else (result, None)

    def fixed_point(update, initial, first=None):
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
            # The initial force evaluation already supplies iteration one.
            iteration = jnp.asarray(max_iter > 0, dtype=jnp.int32)
            value = jnp.where(max_iter > 0, first, initial)
            error = jnp.linalg.norm(value - initial, ord=jnp.inf)
            scale = 1.0 + jnp.linalg.norm(value, ord=jnp.inf)

        return jax.lax.while_loop(
            cond,
            step,
            (iteration, value, error, scale),
        )[1]

    pdot_initial, cache_left = evaluate(pdot, t, x)

    def update_p_half(p_half):
        derivative, _ = evaluate(
            pdot,
            t,
            jnp.concatenate((q, p_half)),
            cache_left,
        )
        return p + 0.5 * dt * derivative

    p_half = fixed_point(update_p_half, p, first=p + 0.5 * dt * pdot_initial)
    qdot_left, _ = evaluate(qdot, t, jnp.concatenate((q, p_half)), cache_left)

    def update_q(q_new):
        qdot_right, _ = evaluate(
            qdot,
            t + dt,
            jnp.concatenate((q_new, p_half)),
        )
        return q + 0.5 * dt * (qdot_left + qdot_right)

    q_new = fixed_point(update_q, q + dt * qdot_left)
    # The final position is the output of the last fixed-point update, not
    # generally the position where its qdot was evaluated. Build a fresh cache.
    pdot_right, _ = evaluate(
        pdot,
        t + dt,
        jnp.concatenate((q_new, p_half)),
    )
    p_new = p_half + 0.5 * dt * pdot_right

    return jnp.concatenate((q_new, p_new))
