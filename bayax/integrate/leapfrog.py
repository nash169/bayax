#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp

from typing import Optional
from bayax.solve import anderson, fixed_point
from bayax.optim.lstsq import lstsq


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


def gen_leapfrog(
    f,
    t,
    x,
    u,
    dt,
    momentum_solver="fixedpoint",
    position_solver="fixedpoint",
    momentum_kwargs: Optional[dict] = None,
    position_kwargs: Optional[dict] = None,
    cache=None,
):
    """
    Advance nonseparable dynamics ``f=(pdot, qdot)`` by one implicit generalized leapfrog step

    momentum_solver finds p_half = p + dt/2 pdot(q, p_half): "fixedpoint" iterates it from the
    explicit predictor; "gauss-newton" minimizes ||r||^2 / 2 for the residual
    r(p_half) = p_half - p - dt/2 pdot(q, p_half) with bayax.solve.lstsq, matrix-free.
    For "gauss-newton", momentum_kwargs must hold linearize(t, x, u, cache=None) returning
    ((jvp, vjp), cache), the products of the momentum Jacobian d pdot / dp at x, from which
    J^T and the curvature J^T J of the residual are built (solved by CG).
    position_solver finds q_new = q + dt/2 (qdot(q, p_half) + qdot(q_new, p_half)) by
    "fixedpoint" or "anderson" iteration. The other momentum_kwargs and position_kwargs
    (tol, max_iter, ...) are passed to the selected solver.

    cache optionally supplies the pdot cache at x, e.g. a geometry already built at q. Callbacks
    may return a derivative or (derivative, cache); the pdot cache at q is reused by the momentum
    solve, linearize and the first qdot. Returns (x_new, cache), with the pdot cache built at
    q_new by the final momentum half step, to pass to the next step. If either solve does not
    converge, x_new is NaN.
    """
    if momentum_solver not in ("fixedpoint", "gauss-newton"):
        raise ValueError('momentum_solver must be "fixedpoint" or "gauss-newton"')
    if position_solver not in ("fixedpoint", "anderson"):
        raise ValueError('position_solver must be "fixedpoint" or "anderson"')
    momentum_kwargs = dict(momentum_kwargs or {})
    position_kwargs = position_kwargs or {}
    pdot, qdot = f
    q, p = jnp.split(x, 2)

    def evaluate(fn, time, state, cache=None):
        result = fn(time, state, u) if cache is None else fn(time, state, u, cache=cache)
        return result if isinstance(result, tuple) else (result, None)

    # solve the momentum equation, reusing the pdot cache at q
    pdot_left, cache = evaluate(pdot, t, x, cache)
    momentum = lambda p_half: evaluate(pdot, t, jnp.concatenate((q, p_half)), cache)[0]
    p_predictor = p + 0.5 * dt * pdot_left
    if momentum_solver == "gauss-newton":
        if "linearize" not in momentum_kwargs:
            raise ValueError('momentum_solver "gauss-newton" requires linearize in momentum_kwargs')
        linearize = momentum_kwargs.pop("linearize")
        residual = lambda p_half: p_half - p - 0.5 * dt * momentum(p_half)

        def linearize_residual(p_half):
            (jvp, vjp), _ = linearize(t, jnp.concatenate((q, p_half)), u, cache=cache)
            jtvp = lambda v: v - 0.5 * dt * vjp(v)
            return jtvp, lambda v: jtvp(v - 0.5 * dt * jvp(v))

        p_half, p_converged, _ = lstsq(residual, linearize_residual, p_predictor, **momentum_kwargs)
    else:
        update_p = lambda p_half: p + 0.5 * dt * momentum(p_half)
        p_half, p_converged = fixed_point(update_p, p, first=p_predictor, **momentum_kwargs)

    # solve the position equation; qdot at q_new needs a fresh cache at each iterate
    qdot_left, _ = evaluate(qdot, t, jnp.concatenate((q, p_half)), cache)
    update_q = lambda q_new: q + 0.5 * dt * (qdot_left + evaluate(qdot, t + dt, jnp.concatenate((q_new, p_half)))[0])
    position = anderson if position_solver == "anderson" else fixed_point
    q_new, q_converged = position(update_q, q + dt * qdot_left, **position_kwargs)

    # explicit final momentum half step at the new position, whose cache starts the next step
    pdot_right, cache_right = evaluate(pdot, t + dt, jnp.concatenate((q_new, p_half)))
    x_new = jnp.concatenate((q_new, p_half + 0.5 * dt * pdot_right))
    return jnp.where(p_converged & q_converged, x_new, jnp.nan), cache_right
