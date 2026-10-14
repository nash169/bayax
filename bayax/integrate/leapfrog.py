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


def generalized_leapfrog(f, t, x, u, dt, tol=1e-10, max_iter=50, reverse_tol=None):
    """
    Advance nonseparable dynamics ``f=(pdot, qdot)`` implicitly

    If either fixed point stops at max_iter without meeting tol, the step
    returns NaN, so the failure propagates through the trajectory and an
    HMC proposal built on it is rejected.

    With reverse_tol, the step is also checked for reversibility: a step
    from the momentum-flipped result must return to the flipped start within
    reverse_tol * (1 + ||x||_inf), otherwise the step returns NaN. This
    catches fixed points converging to a different root in each direction.
    The check assumes time-independent dynamics and doubles the cost.

    Callbacks may return a derivative or (derivative, cache). A returned
    pdot cache must also be accepted by qdot via its cache keyword. Reuse
    this cache only at the same position and time within the step.
    """
    x_new, converged = _generalized_step(f, t, x, u, dt, tol, max_iter)

    if reverse_tol is not None:
        flip = lambda z: jnp.concatenate((z[:z.size // 2], -z[z.size // 2:]))
        x_back, back_converged = _generalized_step(f, t + dt, flip(x_new), u, dt, tol, max_iter)
        error = jnp.linalg.norm(flip(x_back) - x, ord=jnp.inf)
        reversible = error <= reverse_tol * (1.0 + jnp.linalg.norm(x, ord=jnp.inf))
        converged = converged & back_converged & reversible

    return jnp.where(converged, x_new, jnp.nan)


def _generalized_step(f, t, x, u, dt, tol, max_iter):
    """One generalized leapfrog step and whether both fixed points converged."""
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

    pdot_initial, cache_left = evaluate(pdot, t, x)

    def update_p_half(p_half):
        derivative, _ = evaluate(
            pdot,
            t,
            jnp.concatenate((q, p_half)),
            cache_left,
        )
        return p + 0.5 * dt * derivative

    p_half, p_converged = fixed_point(update_p_half, p, first=p + 0.5 * dt * pdot_initial)
    qdot_left, _ = evaluate(qdot, t, jnp.concatenate((q, p_half)), cache_left)

    def update_q(q_new):
        qdot_right, _ = evaluate(
            qdot,
            t + dt,
            jnp.concatenate((q_new, p_half)),
        )
        return q + 0.5 * dt * (qdot_left + qdot_right)

    q_new, q_converged = fixed_point(update_q, q + dt * qdot_left)
    # The final position is the output of the last fixed-point update, not
    # generally the position where its qdot was evaluated. Build a fresh cache.
    pdot_right, _ = evaluate(
        pdot,
        t + dt,
        jnp.concatenate((q_new, p_half)),
    )
    p_new = p_half + 0.5 * dt * pdot_right

    return jnp.concatenate((q_new, p_new)), p_converged & q_converged
