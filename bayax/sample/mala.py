#!/usr/bin/env python
# encoding: utf-8

from jax import log_compiles

from bayax.diffusion.kernels import gaussian_kernel
from bayax.sample import mcmc, mh
from typing import Callable, Optional


def mala(
    f: Callable,
    log_prob: Callable,
    num_steps: int = 10,
    dt: float = 0.01,
    q: Callable = gaussian_kernel,
    u: Optional[Callable] = None,
    seed: int = 0,
):
    def proposal(x, log_q_curr, iter, key):
        # t = iter*dt
        # if log_q_curr is None:
        #     log_q_curr = gaussian_kernel(f, t, x, u, dt)

        # sample proposal
        xn = log_q_curr.sample(key=key)

        # proposal density
        log_q_next = q(f, iter * dt, xn, u, dt)

        return xn, log_q_next, log_q_curr

    return lambda x: mcmc(
        mh(log_prob, proposal, True),
        num_steps=num_steps,
        seed=seed,
    )((x, q(f, 0.0, x, u, dt)))
