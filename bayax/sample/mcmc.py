#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
import jax.random as jr

from typing import Callable, Optional


def mcmc(
    kernel: Callable,
    num_steps: int,
    seed: int = 0,
    aux_kernel: Optional[Callable] = None,
    prob_aux_kernel: float = 0.1
):
    """Run an arbitrary Markov transition kernel.

    With probability prob_aux_kernel a step is taken by aux_kernel instead of
    kernel. Both are selected inside the scan, so they must return the same
    carry and info structures.
    """

    key = jr.key(seed)

    def step(carry, i):
        carry, key = carry

        if aux_kernel is None:
            key, subkey = jr.split(key)
            carry, info = kernel(carry, i, subkey)
        else:
            key, subkey, aux_key = jr.split(key, 3)
            carry, info = jax.lax.cond(
                jr.uniform(aux_key) < prob_aux_kernel,
                lambda _: aux_kernel(carry, i, subkey),
                lambda _: kernel(carry, i, subkey),
                operand=None,
            )

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
