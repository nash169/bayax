#!/usr/bin/env python
# encoding: utf-8

from bayax.diffusion.kernels import gaussian_kernel


def em(f, t, x, u, dt, key, **kwargs):
    dist = gaussian_kernel(f, t, x, u, dt)
    return dist.sample(key=key, **kwargs)


# def em(f, t, x, u, dt, key, **kwargs):
#     drift, diffusion = f(t, x, u)
#     return x + dt * drift + diffusion if drift is not None else x + diffusion
