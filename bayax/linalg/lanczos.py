#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp
import jax.random as jr

from typing import Optional, Callable
from jaxtyping import Array


def _norm_above_tolerance(v, eps):
    """Zero a converged residual before sqrt, keeping its derivative finite."""
    squared = jnp.dot(v, v)
    active = squared > eps**2
    return jnp.where(active, jnp.sqrt(jnp.where(active, squared, 1.0)), 0.0)


def lanczos_tridiag(
    mv: Callable[[Array], Array],
    n: int,
    *,
    m: int = 64,
    eps: float = 1e-12,
    reorth: int = 1,
    v0: Optional[Array] = None,
    key: Optional[Array] = None,
    dtype=jnp.float32,
):
    if v0 is None:
        v0 = jr.normal(key if key is not None else jr.key(0), (n,), dtype=dtype)
    else:
        dtype = v0.dtype
    v0 = v0 / jnp.maximum(jnp.linalg.norm(v0), eps)

    def step(carry, i):
        vecs, beta = carry

        v = vecs[i]
        w = mv(v)
        assert w.shape[0] == n and len(w.shape) == 1, ("Output of mv(v) must be of shape [n].")

        vprev = jax.lax.cond(i > 0, lambda j: vecs[j - 1], lambda _: jnp.zeros_like(v), i)
        w = w - beta * vprev

        alpha = jnp.dot(w, v)
        w = w - alpha * v

        for _ in range(reorth):
            coeffs = vecs @ w
            w = w - vecs.T @ coeffs

        beta = _norm_above_tolerance(w, eps)
        vnext = jnp.where(beta > eps, w / jnp.where(beta > eps, beta, 1.0), jnp.zeros_like(w))

        vecs = jax.lax.cond(i + 1 < m, lambda q: q.at[i + 1].set(vnext), lambda q: q, vecs)

        return (vecs, beta), (alpha, beta)

    (vecs, _), (alpha, beta) = jax.lax.scan(step, (jnp.zeros((m, n), dtype=dtype).at[0].set(
        v0), jnp.array(0.0, dtype=dtype)), jnp.arange(m))
    return alpha, beta[:-1], vecs


def lanczos_bidiag(
    mv: Callable[[Array], Array],
    rmv: Callable[[Array], Array],
    shape: tuple[int, int],
    *,
    m: int = 64,
    eps: float = 1e-12,
    reorth: int = 1,
    v0: Optional[Array] = None,
    key: Optional[Array] = None,
    dtype=jnp.float32,
):
    """
    Golub-Kahan-Lanczos bidiagonalization for rectangular operators.

    Args:
      mv: Maps v -> A v for an operator A with shape ``shape``.
      rmv: Maps u -> A.T u.
      shape: Tuple ``(n_rows, n_cols)`` describing the operator.
      m: Number of bidiagonalization steps.

    Returns:
      alpha: Main diagonal of the bidiagonal matrix, shape ``(m,)``.
      beta: Superdiagonal of the bidiagonal matrix, shape ``(m,)``.
      U: Left Lanczos vectors, shape ``(m, n_rows)``.
      V: Right Lanczos vectors, shape ``(m + 1, n_cols)``.
    """
    n_rows, n_cols = shape

    if v0 is None:
        v0 = jr.normal(key if key is not None else jr.key(0), (n_cols,), dtype=dtype)
    else:
        dtype = v0.dtype
        assert v0.shape == (n_cols,), ("Initial right vector v0 must be of shape [n_cols].")

    assert m > 0, ("m must be positive.")
    v0 = v0 / jnp.maximum(jnp.linalg.norm(v0), eps)

    def step(i, state):
        U, V, alpha, beta = state

        v = V[i]
        p = mv(v)
        assert p.shape[0] == n_rows and len(p.shape) == 1, ("Output of mv(v) must be of shape [n_rows].")
        p = jax.lax.cond(i > 0, lambda x: x - beta[i - 1] * U[i - 1], lambda x: x, p)

        # U only has rows 0..i-1 populated at this point; later rows are zero.
        for _ in range(reorth):
            p = p - U.T @ (U @ p)

        a = _norm_above_tolerance(p, eps)
        u = p * jnp.where(a > eps, 1.0 / jnp.where(a > eps, a, 1.0), 0.0)

        U = U.at[i].set(u)
        alpha = alpha.at[i].set(a.astype(dtype))

        r = rmv(u)
        assert r.shape[0] == n_cols and len(r.shape) == 1, ("Output of rmv(u) must be of shape [n_cols].")
        r = r - a * v

        # V has rows 0..i populated here; later rows are zero.
        for _ in range(reorth):
            r = r - V.T @ (V @ r)

        b = _norm_above_tolerance(r, eps)
        v_next = r * jnp.where(b > eps, 1.0 / jnp.where(b > eps, b, 1.0), 0.0)

        V = V.at[i + 1].set(v_next)
        beta = beta.at[i].set(b.astype(dtype))
        return U, V, alpha, beta

    U, V, alpha, beta = jax.lax.fori_loop(
        0,
        m,
        step,
        (
            jnp.zeros((m, n_rows), dtype=dtype),
            jnp.zeros((m + 1, n_cols), dtype=dtype).at[0].set(v0),
            jnp.zeros((m,), dtype=dtype),
            jnp.zeros((m,), dtype=dtype),
        ),
    )

    return alpha, beta, U, V
