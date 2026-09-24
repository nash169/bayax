#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
import jax.random as jr

from bayax.utils.types import Callable, Optional


def integrate(
    f: Callable | tuple[Callable, Callable],
    integrator: Callable,
    dt: float = 0.01,
    T: float = 1.0,
    u: Optional[Callable] = None,
    seed: Optional[int] = None,
    **kwargs,
):
    key = jr.key(seed) if seed is not None else None

    def step(carry, i):
        t = i * dt
        x, key = carry

        # split key
        if key is not None:
            key, subkey = jr.split(key)

        # integrate
        x = integrator(f=f, t=t, x=x, u=u, dt=dt, **({"key": subkey} if key is not None else {}), **kwargs)
        # x = integrator(f=f if key is None else lambda t, x, u: f(t=t, x=x, u=u, key=subkey), t=t, x=x, u=u, dt=dt)

        return (x, key), (x,)

    return lambda x: jnp.concatenate((jnp.expand_dims(x, axis=0), jax.lax.scan(step, (x, key), jnp.arange(int(round(T / dt))))[1][0]), axis=0)
