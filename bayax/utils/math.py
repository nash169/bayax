from chex import Array, Scalar
import jax
from jax._src.dtypes import dtype
import jax.numpy as jnp

from bayax.utils.types import Callable, Optional, PyTree, Tensor
from bayax.operators.linear_operator import LinearOperator
from bayax.utils.helper import pytree_to_array, array_to_pytree


def hvp(
    f: Callable
) -> Callable:
    def fn(x, v):
        def vjp(x): return jax.lax.map(jax.vjp(f, x)[1], jnp.eye(len(f(x))))
        return jax.jvp(vjp, (x,), (v,))[1]
    return fn


def hvp_2(
    f: Callable
) -> Callable:
    def fn(x, v):
        return jax.jvp(jax.jacrev(f), (x,), (v,))[1]
    return fn

# def hvp(func: Callable, primals, tangents):
#     return jax.jvp(jax.grad(func), (primals,), (tangents,))[1]


def pullback(
    f: Callable,
    h: Optional[LinearOperator] = None
) -> Callable:
    def fn(x, v):
        _, jvp = jax.linearize(f, x)
        hjv = jvp(v) if h is None else h(jvp(v))
        return jax.linear_transpose(jvp, v)(hjv)[0]
    return fn


def inner_jacobian(
    f: Callable,
    h: Optional[LinearOperator] = None
) -> Callable:
    def fn(x, v):
        _, jvp = jax.linearize(f, x)
        hjv = jvp(v) if h is None else h(jvp(v))
        return jax.linear_transpose(jvp, x)(hjv)[0]
    return fn


def outer_jacobian(
    f: Callable,
    h: Optional[LinearOperator] = None
) -> Callable:
    def fn(x, v):
        _, jvp = jax.linearize(f, x)
        jtv = jax.linear_transpose(jvp, x)(v)[0]
        if h is not None:
            leaves, tree_def = jax.tree.flatten(jtv)
            shapes = [leaf.shape for leaf in leaves]
            jtv = array_to_pytree(h(pytree_to_array(jtv)), (shapes, tree_def))
        return jvp(jtv)
    return fn


def gram(
    k: Callable,
    x: Tensor,
    jitter: Scalar = 1e-6
) -> Callable:
    n = x.shape[0]
    k_diag = jax.vmap(lambda i: k(x[i], x[i]))(jnp.arange(n)).squeeze() + jitter
    i_off, j_off = jnp.triu_indices(n, k=1)
    k_off = jax.vmap(lambda i, j: k(x[i], x[j]))(i_off, j_off).squeeze()

    def mv(v):
        y = k_diag * v
        y = y.at[i_off].add(k_off * v[j_off])
        y = y.at[j_off].add(k_off * v[i_off])
        return y

    return mv


def diag_exact(
    mv: Callable,
    n: int
):
    I = jnp.eye(n)
    cols = jax.vmap(mv)(I)
    return jnp.diag(cols)


def diag_hutch(
    mv: Callable,
    n: int,
    key: jax.Array,
    s: int = 20,
):
    def single_sample(key):
        z = jax.random.rademacher(key, (n,), dtype=jnp.float64)
        Az = mv(z)
        return z * Az

    keys = jax.random.split(key, num_samples)
    estimates = jax.vmap(single_sample)(keys)
    return jnp.mean(estimates, axis=0)


def diag_hutchpp(
    mv: Callable,
    n: int,
    key: jax.Array,
    r: int = 100,
    s: int = 100
):
    """
    Hutch++ diagonal estimator for a symmetric matrix A given only A_mv.
    Args:
      A_mv: function (n,) -> (n,), computes A @ v
      n: dimension
      key: PRNGKey
      r: subspace size (captures dominant structure)
      s: Hutchinson samples on the residual
    Returns:
      diag_est: (n,) estimate of diag(A)
    """

    def _orthonormalize(Y, eps=1e-12):
        # QR gives an orthonormal basis for range(Y)
        Q, R = jnp.linalg.qr(Y, mode="reduced")
        # Optional: avoid NaNs if Y is rank-deficient
        diagR = jnp.abs(jnp.diag(R))
        Q = jnp.where(diagR[None, :] > eps, Q, 0.0)
        return Q

    k1, k2 = jax.random.split(key)

    # 1) Build a sketch subspace: Y = A @ G, where G is Gaussian n x r
    G = jax.random.normal(k1, (n, r))
    # Compute Y = A G with vmap over columns
    Y = jax.vmap(mv, in_axes=1, out_axes=1)(G)  # (n, r)
    Q = _orthonormalize(Y)                        # (n, r)

    # 2) Low-rank part: diag(Q (Q^T A Q) Q^T)
    # Compute AQ = A Q
    AQ = jax.vmap(mv, in_axes=1, out_axes=1)(Q)   # (n, r)
    B = Q.T @ AQ                                     # (r, r)  ~ Q^T A Q
    # diag of Q B Q^T: compute rowwise dot: sum_j Q_ij * (Q B)_ij
    QB = Q @ B                                       # (n, r)
    diag_lowrank = jnp.sum(Q * QB, axis=1)           # (n,)

    # 3) Residual part via Hutchinson on (I - QQ^T) A (I - QQ^T)
    Z = jax.random.rademacher(k2, (n, s))                # (n, s)
    # Project: W = (I - QQ^T) Z
    QtZ = Q.T @ Z                                    # (r, s)
    W = Z - Q @ QtZ                                  # (n, s)
    # Apply A: AW = A W
    AW = jax.vmap(mv, in_axes=1, out_axes=1)(W)    # (n, s)
    # Project again: (I - QQ^T) AW
    QtAW = Q.T @ AW
    R = AW - Q @ QtAW
    # Hutchinson diag estimate: mean over samples of w ⊙ r
    diag_resid = jnp.mean(W * R, axis=1)

    return diag_lowrank + diag_resid


def bandwidth(x, y=None, reduce_fn=jnp.median):
    from bayax.kernels.rbf import rbf
    k_rbf = jax.jit(
        jax.vmap(
            jax.vmap(rbf, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None),
        )
    )

    z = x if y is None else jnp.concatenate([x, y], axis=0)
    d2 = -2 * jnp.log(k_rbf(z, z, 0.0, 0.0))

    return jnp.sqrt(0.5 * reduce_fn(d2[d2 > 0.0]))
