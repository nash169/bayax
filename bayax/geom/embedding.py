#!/usr/bin/env python
# encoding: utf-8

import jax
import jax.numpy as jnp

from bayax.utils.types import Callable, Optional
from bayax.linalg.svd import svd
from bayax.operators.linear_operator import LinearOperator
from bayax.operators import LowRankOperator


# h is a squared root factor L of the ambient metric H = L L^T throughout this module.
# Vector arguments may be matrices, whose column contractions are summed; u defaults to v.

def _ambient(h, w):
    # H w = L L^T w
    return h(h.transpose()(w)) if h is not None else w


def _columns(fn, w):
    # apply a vector map to a vector, or column-wise to a matrix
    return fn(w) if w.ndim == 1 else jax.vmap(fn, in_axes=1, out_axes=1)(w)


def _summed(fn, v, u):
    # fn(v, u) for vectors, summed over paired columns for matrices; u=None pairs v with itself
    if v.ndim == 1:
        return fn(v, u)
    return jax.vmap(fn, in_axes=(1, None if u is None else 1), out_axes=1)(v, u).sum(axis=1)


def pullmetric(
    f: Callable,
    h: Optional[LinearOperator] = None
) -> Callable:
    r"""
    Metric-vector product of the pullback metric G = J^T H J

    Returns fn(x, v), from one linearization of f at x.
    """
    def fn(x, v):
        J = jax.linearize(f, x)[1]
        JT = jax.linear_transpose(J, x)
        return _columns(lambda c: JT(_ambient(h, J(c)))[0], v)
    return fn


def pullmetric_grad(
    f: Callable,
    h: Optional[LinearOperator] = None
) -> Callable:
    r"""
    Directional derivative of the pullback metric (DG[v]) u = J^T H D^2f[v, u] + D^2f[v, .]^T H J u

    Returns fn(x, v, u=None). One linearization of z -> (f(z), J(z) v) gives the joint map
    w -> (J w, D^2f[v, w]), whose transpose applies both terms at once.
    """
    def fn(x, v, u=None):
        def single(v, u):
            _, joint = jax.linearize(lambda z: jax.jvp(f, (z,), (v,)), x)
            ju, hess = joint(v if u is None else u)
            return jax.linear_transpose(joint, x)((_ambient(h, hess), _ambient(h, ju)))[0]
        return _summed(single, v, u)
    return fn


def christoffel_fk(
    f: Callable,
    h: Optional[LinearOperator] = None,
    contraction="ij"
) -> Callable:
    r"""
    Contract first-kind Christoffel symbols T_kij = <d_k f, d_i d_j f>_H of the pullback metric.

    Returns fn(x, v, u=None), with v and u held fixed in position derivatives.
    For "ij": sum_ij T_kij v_i u_j = J^T H D^2f[v, u].
    For "ki": sum_ki T_kij v_k u_i = D^2f[u, .]^T H J v.
    """
    if contraction == "ij":
        def fn(x, v, u=None):
            def single(v, u):
                u = v if u is None else u
                hess = jax.jvp(lambda z: jax.jvp(f, (z,), (u,))[1], (x,), (v,))[1]
                return jax.vjp(f, x)[1](_ambient(h, hess))[0]
            return _summed(single, v, u)
    elif contraction == "ki":
        def fn(x, v, u=None):
            # matrices: map J over columns only, so the pullback to x runs once on the summed cotangent
            u = v if u is None else u
            Ju = lambda z: _columns(lambda c: jax.jvp(f, (z,), (c,))[1], u)
            ju, pullback = jax.vjp(Ju, x)
            jv = ju if v is u else _columns(lambda c: jax.jvp(f, (x,), (c,))[1], v)
            return pullback(_ambient(h, jv))[0]
    else:
        raise ValueError('contraction must be "ij" or "ki"')
    return fn


def christoffel_sk(
    f: Callable,
    h: Optional[LinearOperator] = None,
    g_inv: Optional[Callable] = None,
    contraction="ij",
) -> Callable:
    r"""
    Contract second-kind Christoffel symbols T^k_ij of the pullback metric.

    g_inv(x, v) applies the inverse metric to a vector; by default CG on pullmetric.
    Returns fn(x, v, u=None).
    For "ij": sum_ij T^k_ij v^i u^j, applying G^-1 to the first-kind output.
    For "ki": sum_ki T^k_ij v_k u^i, applying G^-1 to the covector v; u defaults to G^-1 v,
    so with v = p this is the geodesic force of the momentum.
    """
    fk = christoffel_fk(f, h, contraction=contraction)
    if g_inv is None:
        g = pullmetric(f, h)

        def g_inv(x, v):
            return jax.scipy.sparse.linalg.cg(lambda w: g(x, w), v)[0]

    if contraction == "ij":
        def fn(x, v, u=None):
            return g_inv(x, fk(x, v, u))
    else:
        def fn(x, v, u=None):
            return fk(x, _columns(lambda c: g_inv(x, c), v), u)
    return fn


@jax.tree_util.register_pytree_node_class
class EmbeddingGeom:
    r"""
    Embedding geometry whose pytree children are arrays only, with f held as static aux data.

    Products with J are recomputed from f at x through the functions of this module, so two
    instances built from the same f object share a treedef: they can be carried through scan,
    while_loop or cond, and passed to jit without retracing. f is compared by identity and must
    not close over traced values.
    """

    def __init__(
        self,
        f: Callable,
        x: jax.Array,
        h: Optional[LinearOperator] = None,
        eps: float = 1e-8,
        jitter: Optional[float] = None,
        lowrank: bool = False,
        **kwargs
    ):
        # h is a squared root factor L of the ambient metric H = L L^T
        self.f = f
        self.x = x
        self.h = h
        self.jitter = jitter
        self.lowrank = lowrank

        # lowrank: replace M = J^T L by U S from its SVD and store metric as low rank operator
        self.metric = None
        if lowrank:
            shape = (x.size, jax.eval_shape(self._jvp, x).shape[0])
            u, s, _ = svd(self._jtvp, shape=shape, rmv=self._jvp, dtype=x.dtype, **kwargs)
            self.metric = LowRankOperator(sval=s**2, left=u, eps=eps, jimg=jitter, jker=jitter)

    def tree_flatten(self):
        return (self.x, self.h, self.jitter, self.metric), (self.f, self.lowrank)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls.__new__(cls)
        obj.x, obj.h, obj.jitter, obj.metric = children
        obj.f, obj.lowrank = aux
        return obj

    def _jvp(self, v):
        # L^T J v
        jv = jax.jvp(self.f, (self.x,), (v,))[1]
        return self.h.transpose()(jv) if self.h is not None else jv

    def _jtvp(self, w):
        # J^T L w
        return jax.vjp(self.f, self.x)[1](self.h(w) if self.h is not None else w)[0]

    def jvp(self, v):
        # lowrank: S U^T v, with S and U held by the metric
        return self.metric._sqrt_sval() * (self.metric.left.T @ v) if self.lowrank else self._jvp(v)

    def jtvp(self, w):
        # lowrank: U S w, with S and U held by the metric
        return self.metric.left @ (self.metric._sqrt_sval() * w) if self.lowrank else self._jtvp(w)

    def metric_mv(self, v):
        if self.lowrank:
            return self.metric @ v
        gv = pullmetric(self.f, self.h)(self.x, v)
        return gv + self.jitter * v if self.jitter is not None else gv

    def metric_solve(self, v, **kwargs):
        if self.lowrank:
            return self.metric.solve(v)
        return jax.scipy.sparse.linalg.cg(self.metric_mv, v, **kwargs)[0]

    def metric_logdet(self):
        if self.lowrank:
            return self.metric.logdet()
        raise NotImplementedError("metric_logdet requires lowrank=True.")

    def cfk_mv(self, u, v=None, contraction="ij"):
        r"""
        Contract first-kind Christoffel symbols T_kij = <d_k f, d_i d_j f>_H of the embedding metric.

        u and v are vectors, or matrices whose column contractions are summed; v defaults to u.
        Both channels differentiate the exact pullback metric J^T H J.
        For "ij": sum_ij T_kij v_i u_j = J^T H D^2f[v, u].
        For "ki": sum_ki T_kij v_k u_i = D^2f[u, .]^T H J v.
        """
        fn = christoffel_fk(self.f, self.h, contraction=contraction)
        return fn(self.x, u) if v is None else fn(self.x, v, u)

    def metric_grad_mv(self, v, u=None):
        r"""
        Directional derivative of the metric (DG[v]) u = J^T H D^2f[v, u] + D^2f[v, .]^T H J u,
        summed over columns for matrices; u defaults to v.

        Both channels differentiate the exact pullback metric J^T H J.
        """
        return pullmetric_grad(self.f, self.h)(self.x, v, u)

    def metric_inv_grad_mv(self, v, u=None, **kwargs):
        r"""
        Directional derivative of the inverse metric (DG^-1[v]) u = -G^-1 (DG[v]) G^-1 u,
        summed over columns for matrices; u defaults to v.

        G^-1 is metric_solve (kwargs are passed to it) and DG is metric_grad_mv of the exact pullback
        metric J^T H J. With lowrank=True this matches the derivative of the low rank inverse only
        where the low rank metric equals the exact one (no truncation of nonzero modes).
        """
        u = v if u is None else u
        solve = lambda w: self.metric_solve(w, **kwargs)
        # the outer G^-1 is linear, so it is applied once to the column sum
        return -solve(self.metric_grad_mv(v, _columns(solve, u)))

    def half_grad_logdet(self):
        r"""
        Half gradient of metric_logdet, sum_i (lambda_i + jimg)^-1 u_i^T DG[.] u_i / 2 over retained modes.

        The retained eigenpairs of the exact metric J^T H J satisfy d lambda_i = u_i^T dG u_i, so only
        a "ki" contraction with the retained modes is needed, without differentiating the SVD.
        Thresholded and complement modes enter metric_logdet as constants and drop out.
        Requires distinct eigenvalues at the truncation cut.
        """
        if not self.lowrank:
            raise NotImplementedError("half_grad_logdet requires lowrank=True.")
        metric = self.metric
        d = metric.sval if metric.jimg is None else metric.sval + metric.jimg
        scale = jnp.where(metric.ker, 0.0, jax.lax.rsqrt(jnp.where(metric.ker, 1.0, d)))
        return self.cfk_mv(metric.left * scale, contraction="ki")
