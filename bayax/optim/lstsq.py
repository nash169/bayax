import jax
import jax.numpy as jnp

from typing import Callable, Optional
from jax.scipy.sparse.linalg import cg

from bayax.operators import LowRankOperator


def lstsq(
    r: Callable,
    linearize: Callable,
    x0: jnp.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10,
    rtol: Optional[float] = None,
    damping: Optional[float] = None,
    max_damping_iter: int = 20,
    cg_kwargs: Optional[dict] = None,
):
    r"""
    Minimize 0.5 ||r(x)||^2 by steps (H + mu I) dx = -J^T r

    linearize(x) returns (jtvp, hvp): J^T as an array or callable, and the
    curvature H as a LowRankOperator (closed-form inverse; pseudo-inverse if
    jimg = jker = None), an array (direct solve) or a callable (CG). J^T J
    gives Gauss-Newton, the full Hessian Newton.

    With damping, Levenberg-Marquardt: mu starts at damping and follows
    Nielsen's gain-ratio update. linearize is called once per accepted step:
    a rejected step is re-solved with larger mu at the same linearization,
    up to max_damping_iter times, after which the solve stops unconverged.

    Stops when ||dx|| <= tol, when ||r||_inf <= rtol * (1 + ||x||_inf), or
    after max_iter steps. Returns (x, converged, r(x)).
    """
    def apply(A, v):
        return A @ v if isinstance(A, jax.Array) else A(v)

    def solve(H, b, mu):
        if isinstance(H, LowRankOperator):
            if damping is not None:
                H = LowRankOperator(sval=H.sval, left=H.left, eps=H.eps, jimg=mu, jker=mu)
            return H.inverse()(b)
        if isinstance(H, jax.Array):
            return jnp.linalg.solve(H + mu * jnp.eye(H.shape[0], dtype=H.dtype), b)
        return cg(lambda v: H(v) + mu * v, b, **(cg_kwargs or {}))[0]

    def small_residual(x, residual):
        if rtol is None:
            return jnp.array(False)
        return jnp.linalg.norm(residual, ord=jnp.inf) <= rtol * (1.0 + jnp.linalg.norm(x, ord=jnp.inf))

    def cond(carry):
        x, residual, _, _, iteration, step_norm = carry
        return (iteration < max_iter) & (step_norm > tol) & ~small_residual(x, residual)

    def step(carry):
        x, residual, mu, nu, iteration, _ = carry
        jtvp, hvp = linearize(x)
        g = apply(jtvp, residual)

        def trial(inner):
            _, _, _, mu, nu, tries = inner
            dx = solve(hvp, -g, mu)
            residual_new = r(x + dx)
            if damping is None:
                return dx, residual_new, jnp.array(True), mu, nu, tries + 1
            # gain ratio: actual over predicted reduction, 0.5 dx^T (mu dx - g)
            rho = (residual @ residual - residual_new @ residual_new) / (dx @ (mu * dx - g))
            accept = rho > 0
            mu = jnp.where(accept, mu * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3), mu * nu)
            nu = jnp.where(accept, 2.0, 2.0 * nu)
            return dx, residual_new, accept, mu, nu, tries + 1

        dx, residual_new, accept, mu, nu, _ = jax.lax.while_loop(
            lambda inner: ~inner[2] & (inner[5] < max_damping_iter),
            trial,
            (jnp.zeros_like(x), residual, jnp.array(False), mu, nu, jnp.array(0)),
        )
        # no accepted step: NaN step norm ends the loop and reports no convergence
        return (
            jnp.where(accept, x + dx, x),
            jnp.where(accept, residual_new, residual),
            mu, nu, iteration + 1,
            jnp.where(accept, jnp.linalg.norm(dx), jnp.nan),
        )

    mu0 = jnp.asarray(0.0 if damping is None else damping, dtype=x0.dtype)
    x, residual, _, _, _, step_norm = jax.lax.while_loop(
        cond,
        step,
        (x0, r(x0), mu0, jnp.asarray(2.0, dtype=x0.dtype), jnp.array(0), jnp.array(jnp.inf, dtype=x0.dtype)),
    )
    converged = ((step_norm <= tol) | small_residual(x, residual)) & jnp.all(jnp.isfinite(x))
    return x, converged, residual
