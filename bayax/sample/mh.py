#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
import jax.random as jr

from typing import Callable


def mh(
    log_p: Callable,  # log target probability
    proposal: Callable,  # log proposal probabilties
    carry_proposal: bool = True
):
    """
    proposal(x, iteration, key) returns:
        log q(y | x)
        log q(x | y)
    """

    def kernel(carry, iter, key):
        x = carry[0]
        pkey, akey = jr.split(key)

        xn, log_q_next, log_q_curr = proposal(*carry, iter, pkey)

        log_alpha = jnp.minimum(0.0, log_p(xn) + log_q_next(x) - log_p(x) - log_q_curr(xn))
        accept = jnp.log(jax.random.uniform(akey, dtype=x.dtype)) < log_alpha

        carry = jax.lax.cond(
            accept,
            lambda _: (xn, log_q_next) if carry_proposal else (xn,),
            lambda _: carry,
            operand=None,
        )

        info = {
            "accepted": accept,
            "log_alpha": log_alpha,
        }

        return carry, info

    return kernel
