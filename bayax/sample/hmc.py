#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
import jax.random as jr

from bayax.integrate.integrate import integrate
from bayax.integrate import leapfrog, gen_leapfrog
from bayax.diffusion.hamiltonian import d2Hdqdp_geom
from bayax.sample.mcmc import mcmc
from typing import Callable, Optional


def _reverse_check(trajectory, q, p, qn, pn, tol):
    """
    Return (qn, pn) if the trajectory from (qn, -pn) returns to (q, -p), else NaN

    The return must satisfy ||x_back - x||_inf <= tol * (1 + ||x||_inf) with x = (q, -p).
    This catches implicit solves converging to a different root in each direction; it
    assumes time-independent dynamics and doubles the integration cost.
    """
    x = jnp.concatenate((q, -p))
    x_back = trajectory(jnp.concatenate((qn, -pn)))[-1]
    reversible = jnp.linalg.norm(x_back - x, ord=jnp.inf) <= tol * (1.0 + jnp.linalg.norm(x, ord=jnp.inf))
    return jnp.where(reversible, qn, jnp.nan), jnp.where(reversible, pn, jnp.nan)


def hmc(
    H: Callable,
    p_pdf: Callable,
    num_steps: int = 10,
    dt: float = 0.01,
    T: float = 1.0,
    nsh: bool = False,
    dH: Optional[tuple[Callable, Callable]] = None,
    geom: Optional[Callable] = None,
    u: Optional[Callable] = None,
    seed: int = 0,
    aux_kernel: Optional[Callable] = None,
    prob_aux_kernel: float = 0.1,
    reverse_check: bool = False,
    reverse_tol: float = 1e-6,
    **kwargs,
):
    """
    Sample positions using Hamiltonian dynamics

    H may return a scalar or (value, cache). Supplied dH contains the signed
    dynamics (pdot, qdot) = (-dH/dq, dH/dp); generalized leapfrog also accepts
    dynamics returning (derivative, cache). A proposal with non-finite
    energy, e.g. from an unconverged generalized leapfrog step, is flagged
    as divergent and rejected.

    With nsh, the generalized leapfrog gen_leapfrog integrates the dynamics and kwargs
    configures it (momentum_solver, position_solver, momentum_kwargs, position_kwargs).
    For momentum_solver="gauss-newton" without a linearize in momentum_kwargs, geom(q)
    builds it with d2Hdqdp_geom. The pdot cache returned by each step starts the next one, so a
    geometry in it must keep a fixed pytree structure: build it with bayax.geom.embedding.

    With reverse_check, each trajectory is integrated back from its momentum-flipped end and
    must return to the flipped start within reverse_tol (see _reverse_check); otherwise the
    proposal is flagged as divergent and rejected.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")

    def energy(t, x, u):
        result = H(t, x, u)
        return result[0] if isinstance(result, tuple) else result

    if dH is None:
        def qdot(t, x, u):
            q, p = jnp.split(x, 2)
            return jax.grad(lambda t, q, p, u: energy(t, jnp.concat([q, p]), u), argnums=2)(t, q, p, u)

        def pdot(t, x, u):
            q, p = jnp.split(x, 2)
            return -jax.grad(lambda t, q, p, u: energy(t, jnp.concat([q, p]), u), argnums=1)(t, q, p, u)
    else:
        pdot, qdot = dH

    momentum_kwargs = kwargs.get("momentum_kwargs") or {}
    if nsh and kwargs.get("momentum_solver") == "gauss-newton" and "linearize" not in momentum_kwargs:
        if geom is None:
            raise ValueError('momentum_solver "gauss-newton" requires geom or linearize in momentum_kwargs')
        d2Hdqdp = d2Hdqdp_geom(geom)

        def linearize(t, x, u, cache=None):
            # pdot = -dH/dq, so its momentum derivative is -d2H/dqdp
            (jvp, vjp), cache = d2Hdqdp(t, x, u, cache=cache)
            return (lambda v: -jvp(v), lambda v: -vjp(v)), cache

        kwargs["momentum_kwargs"] = {**momentum_kwargs, "linearize": linearize}

    integrate_trajectory = integrate(
        f=(pdot, qdot),
        integrator=gen_leapfrog if nsh else leapfrog,
        dt=dt,
        T=T,
        u=u,
        **kwargs,
    )

    def kernel(carry, iter, key):
        momentum_key, accept_key = jr.split(key)
        q, pdf_curr = carry

        p = pdf_curr.sample(key=momentum_key)
        qn, pn = jnp.split(integrate_trajectory(jnp.concatenate((q, p)))[-1], 2)
        if reverse_check:
            qn, pn = _reverse_check(integrate_trajectory, q, p, qn, pn, reverse_tol)

        pdf_next = p_pdf(qn)
        H_curr = energy(None, jnp.concat([q, p]), u)
        H_next = energy(None, jnp.concat([qn, pn]), u)

        divergent = ~jnp.isfinite(H_next)
        log_alpha = jnp.where(divergent, -jnp.inf, jnp.minimum(0.0, H_curr - H_next))
        accept = jnp.log(jr.uniform(accept_key, dtype=q.dtype)) < log_alpha

        carry = jax.lax.cond(
            accept,
            lambda _: (qn, pdf_next),
            lambda _: carry,
            operand=None,
        )

        info = {
            "accepted": accept,
            "log_alpha": log_alpha,
            "divergent": divergent,
        }
        return carry, info

    return lambda x: mcmc(
        kernel,
        num_steps=num_steps,
        seed=seed,
        aux_kernel=aux_kernel,
        prob_aux_kernel=prob_aux_kernel
    )((x, p_pdf(x)))
