#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
import jax.random as jr

from bayax.integrate.integrate import integrate
from bayax.integrate import leapfrog, generalized_leapfrog
from bayax.sample.mcmc import mcmc
from typing import Callable, Optional


def hmc(
    H: Callable,
    p_pdf: Callable,
    num_steps: int = 10,
    dt: float = 0.01,
    T: float = 1.0,
    nsh: bool = False,
    dH: Optional[tuple[Callable, Callable]] = None,
    u: Optional[Callable] = None,
    seed: int = 0,
    **kwargs,
):
    """
    Sample positions using Hamiltonian dynamics

    H may return a scalar or (value, cache). Supplied dH contains the signed
    dynamics (pdot, qdot) = (-dH/dq, dH/dp); generalized leapfrog also accepts
    dynamics returning (derivative, cache).
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

    integrate_trajectory = integrate(
        f=(pdot, qdot),
        integrator=generalized_leapfrog if nsh else leapfrog,
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

        pdf_next = p_pdf(qn)
        H_curr = energy(None, jnp.concat([q, p]), u)
        H_next = energy(None, jnp.concat([qn, pn]), u)

        log_alpha = jnp.minimum(0.0, H_curr - H_next)
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
        }
        return carry, info

    return lambda x: mcmc(
        kernel,
        num_steps=num_steps,
        seed=seed,
    )((x, p_pdf(x)))
