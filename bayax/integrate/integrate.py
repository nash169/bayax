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
    """
    Return x -> trajectory of the integrator, with x as its first row

    The integrator may return x or (x, cache); a returned cache is passed to the next step as its
    cache keyword. The cache must keep a fixed pytree structure across steps, as geometries from
    bayax.geom.embedding do.
    """
    key = jr.key(seed) if seed is not None else None

    def step(carry, i):
        t = i * dt
        x, cache, key = carry

        # split key
        if key is not None:
            key, subkey = jr.split(key)

        # integrate, starting from the cache returned by the previous step
        x = integrator(f=f, t=t, x=x, u=u, dt=dt, **({"key": subkey} if key is not None else {}),
                       **({"cache": cache} if cache is not None else {}), **kwargs)
        # x = integrator(f=f if key is None else lambda t, x, u: f(t=t, x=x, u=u, key=subkey), t=t, x=x, u=u, dt=dt)
        x, cache = x if isinstance(x, tuple) else (x, None)

        return (x, cache, key), (x,)

    def scan_step(carry, i):
        new_carry, out = step(carry, i)
        if jax.tree_util.tree_structure(new_carry[1]) != jax.tree_util.tree_structure(carry[1]):
            raise TypeError(
                "The integrator cache changes its pytree structure between steps, so it cannot be "
                "carried; use a geometry with a fixed structure, e.g. bayax.geom.embedding."
            )
        return new_carry, out

    def run(x):
        num_steps = int(round(T / dt))
        if num_steps == 0:
            return jnp.expand_dims(x, axis=0)
        # the first step runs outside the scan, so its cache fixes the carry structure
        carry, (x_first,) = step((x, None, key), 0)
        xs = jax.lax.scan(scan_step, carry, jnp.arange(1, num_steps))[1][0]
        return jnp.concatenate((jnp.expand_dims(x, axis=0), jnp.expand_dims(x_first, axis=0), xs), axis=0)

    return run
