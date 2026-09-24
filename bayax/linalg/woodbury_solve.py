#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
import jax.numpy as jnp

from bayax.utils.types import Vector, Matrix, Scalar


def woodbury_solve(
    U: Matrix,
    s: Vector,
    d: Scalar | Vector,
    v: Vector
):
    """
    A = U S U^T + D

    Args
      U: PSD Eigenvectors.
      s: PSD Eigenvalues, S = diag{s}
      d: Diagonal matrix, D = diag{d}.
      v: Vector.

    Returns:
      Solution of the linear system.
    """
    if isinstance(d, Vector):
        D_inv_v = v / d
        D_inv_U = U / d[:, None]
        return D_inv_v - D_inv_U @ jax.scipy.linalg.cho_solve((jnp.linalg.cholesky((U.T @ D_inv_U).at[jnp.diag_indices(len(s))].add(1 / s)), True), U.T @ D_inv_v)
    else:
        return v / d - (U / d / (d / s + 1)) @ (U.T @ v)


def woodbury_chol_solve(
    L: Matrix,
    d: Scalar | Vector,
    v: Vector
):
    """
    A = L L^T + D
    A^{-1} b = D^{-1} v - D^{-1} L (I + L^T D^{-1} L)^{-1} L^T D^{-1} v

    Args
      L: Low-rank approximation of PSD.
      d: Diagonal matrix, D = diag{d}.
      v: Vector.

    Returns:
      Solution of the linear system.
    """
    D_inv_v = v / d
    D_inv_L = L / (d[:, None] if isinstance(d, Vector) else d)
    eye = jnp.eye(L.shape[-1])
    return D_inv_v - D_inv_L @ jax.scipy.linalg.cho_solve((jnp.linalg.cholesky(eye + L.T @ D_inv_L), True), L.T @ D_inv_v)
