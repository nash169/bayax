#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
from jaxtyping import Array


def svd(op, *, shape=None, rmv=None, k=None, method='dense', dtype=None, **kwargs):
    """Reduced SVD of an array or linear map; returns U, s, V.T.

    ``dense`` materializes callable operators (``svd`` is an alias). ``randomized`` accepts
    oversampling (default 10), n_iter (default 2), and key. ``lanczos``
    accepts m (subspace size), key, and lanczos_bidiag options.
    Iterative methods require k. Keep key fixed when differentiating a
    position-dependent approximation; derivatives describe the approximation.
    """
    if callable(op):
        if shape is None:
            raise ValueError('shape is required for a callable SVD')
        mv = op
        dtype = jnp.asarray(0.).dtype if dtype is None else dtype
        if rmv is None:
            transpose = jax.linear_transpose(mv, jnp.zeros(shape[1], dtype=dtype))
            rmv = lambda v: transpose(v)[0]
    else:
        shape, dtype = op.shape, op.dtype
        mv, rmv = lambda v: op @ v, lambda v: op.T @ v
    limit = min(shape)
    if k is not None and not 1 <= k <= limit:
        raise ValueError('k must be between 1 and min(shape)')

    if method in ('dense', 'svd'):
        matrix = jax.vmap(mv, in_axes=1, out_axes=1)(jnp.eye(shape[1], dtype=dtype)) if callable(op) else op
        if kwargs.pop('full_matrices', False):
            raise ValueError('lowrank requires full_matrices=False')
        if not kwargs.pop('compute_uv', True):
            raise ValueError('lowrank requires compute_uv=True')
        left, values, right_t = jnp.linalg.svd(matrix, full_matrices=False, **kwargs)
        return left[:, :k], values[:k], right_t[:k]

    if method not in ('randomized', 'lanczos'):
        raise ValueError("method must be 'dense', 'randomized', or 'lanczos'")
    if k is None:
        raise ValueError('matrix-free SVD requires an explicit rank k')
    key = kwargs.pop('key', jax.random.key(0))
    if method == 'lanczos':
        from bayax.linalg.lanczos import lanczos_bidiag
        m = kwargs.pop('m', min(limit, max(2 * k, k + 10)))
        if not k <= m <= limit:
            raise ValueError('m must be between k and min(shape)')
        alpha, beta, U, V = lanczos_bidiag(
            mv, rmv, shape, m=m, key=key, dtype=dtype, **kwargs,
        )
        values, left, right = _svd_lanczos(k, alpha, beta, U, V)
        return left, values, right.T

    oversampling = kwargs.pop('oversampling', 10)
    n_iter = kwargs.pop('n_iter', 2)
    if kwargs:
        raise TypeError(f'Unexpected randomized SVD options: {tuple(kwargs)}')
    if oversampling < 0 or n_iter < 0:
        raise ValueError('oversampling and n_iter must be nonnegative')
    width = min(limit, k + oversampling)
    apply = lambda fn, V: jax.vmap(fn, in_axes=1, out_axes=1)(V)
    omega = jax.random.normal(key, (shape[1], width), dtype=dtype)
    Q = jnp.linalg.qr(apply(mv, omega), mode='reduced')[0]
    for _ in range(n_iter):
        P = jnp.linalg.qr(apply(rmv, Q), mode='reduced')[0]
        Q = jnp.linalg.qr(apply(mv, P), mode='reduced')[0]
    left, values, right_t = jnp.linalg.svd(apply(rmv, Q).T, full_matrices=False)
    return Q @ left[:, :k], values[:k], right_t[:k]


def _svd_lanczos(
    k: int,
    alpha: Array,
    beta: Array,
    U: Array,
    V: Array,
):
    """
    Recover approximate singular triplets from Lanczos bidiagonalization.

    The bidiagonalization stores ``m`` left vectors in ``U`` and ``m + 1``
    right vectors in ``V``. Retain the final residual column in the
    rectangular projected matrix; dropping it loses a mode for wide maps.
    """
    B = jnp.zeros((alpha.size, alpha.size + 1), dtype=alpha.dtype)
    rows = jnp.arange(alpha.size)
    B = B.at[rows, rows].set(alpha).at[rows, rows + 1].set(beta)
    Ub, s, Vhb = jnp.linalg.svd(B, full_matrices=False)

    idx = jnp.argsort(s, descending=True)[:k]
    s = s[idx]

    left = U.T @ Ub[:, idx]
    right = V.T @ Vhb.T[:, idx]

    left = left / jnp.maximum(jnp.linalg.norm(left, axis=0, keepdims=True), 1e-12)
    right = right / jnp.maximum(jnp.linalg.norm(right, axis=0, keepdims=True), 1e-12)

    return s, left, right
