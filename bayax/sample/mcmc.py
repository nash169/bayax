#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
import jax.random as jr

from typing import Callable


def mcmc(
    kernel: Callable,
    num_steps: int,
    seed: int = 0,
):
    """Run an arbitrary Markov transition kernel."""

    key = jr.key(seed)

    def step(carry, i):
        carry, key = carry
        key, subkey = jr.split(key)

        carry, info = kernel(carry, i, subkey)

        return (carry, key), (carry[0], info)

    def run(carry):
        (_, _), (xs, infos) = jax.lax.scan(
            step,
            (carry, key),
            jnp.arange(num_steps),
        )

        xs = jnp.concatenate((carry[0][None, ...], xs), axis=0)
        return xs, infos

    return run
