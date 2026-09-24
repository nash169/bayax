from typing import Callable

from bayax.densities.abstract_density import ProjectedDensity
from bayax.densities import MultivariateNormal
from bayax.operators import ScaledOperator
from bayax.linalg.project import project_operator


def gaussian_kernel(f, t, x, u, dt):
    drift, cov = f(t=t, x=x, u=u)
    return MultivariateNormal(cov=ScaledOperator(dt, cov), mean=x if drift is None else x + dt * drift)


def projected_kernel(kernel_fn: Callable, lift: Callable, project: Callable) -> Callable:
    return project_operator(kernel_fn, lift, lambda density, x: ProjectedDensity(density, lift, project, x), "x")
