# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import os
import sys
import time
import glob
import gc

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
jax.config.update('jax_enable_x64', True)


# %matplotlib widget
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times'

sys.path.insert(0, os.path.abspath(os.path.join('../')))

# %% [markdown]
# ## Find MAP

# %%
from bayax.utils import DataLoader

num_samples_1, num_samples_2 = 100, 100
samples_c1 = jax.random.multivariate_normal(jax.random.key(0), jnp.array([-1, -1]), 0.2 * jnp.eye(2), shape=(num_samples_1,))
targets_c1 = jnp.zeros(num_samples_1)
samples_c2 = jax.random.multivariate_normal(jax.random.key(0), jnp.array([1, 1]), 0.2 * jnp.eye(2), shape=(num_samples_2,))
targets_c2 = jnp.ones(num_samples_2)

samples = jnp.concatenate((samples_c1, samples_c2), axis=0)
targets = jnp.concatenate((targets_c1, targets_c2))
train_loader = DataLoader(samples, targets, 30, shuffle=True)

# %%
from bayax.models import MLP
model = MLP(
    layers=[2, 1],
    use_bias=False,
    param_dtype=jax.numpy.float64
)
num_params = model.size

# %%
from bayax.densities import Bernoulli, MultivariateNormal
from bayax.functions.likelihood import neg_logll
from bayax.operators import DiagOperator

p_ll = lambda f: Bernoulli(f, logits=True)
prior_var = DiagOperator(
    diag=jnp.array(10.),
    dim=num_params
)
p_prior = MultivariateNormal(cov=prior_var)

# %%
import optax
from flax import nnx

n_epochs = 1000
step_size = 1e-3
optimizer = nnx.Optimizer(model, optax.adam(step_size))


def loss_fn(model, x, y):
    y_pred = model(x)
    return neg_logll(p_ll, y, y_pred) - p_prior(model.params) / y.shape[0]


@nnx.jit
def train_step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(grads)
    return loss


# %%
losses = []
for epoch in range(n_epochs):
    for x_tr, y_tr in train_loader:
        losses.append(train_step(model, optimizer, x_tr, y_tr))

    if epoch % 100 == 0:
        print(f"[epoch {epoch}]: loss: {losses[-1]:.4f}")

print(f'{optimizer.step.value=}')
print(f"Final loss: {losses[-1]:.4f}")

# %%
from bayax.utils.plot import contour_plot, surf_plot
fig = contour_plot(lambda x: jax.nn.sigmoid(model(x)), min=[-3, -3], max=[3, 3], res=100, iso=False, alpha=0.5, zorder=-1, label='\sigma(f(X))')
ax = fig.axes[0]
ax.scatter(samples_c1[:, 0], samples_c1[:, 1], label='class 1', color='green', alpha=0.5)
ax.scatter(samples_c2[:, 0], samples_c2[:, 1], label='class 2', color='orange', alpha=0.5)
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_aspect('equal', 'box')
ax.legend()


# %%
def loss(params):
    return neg_logll(p_ll, targets, model.fwd_params()(samples, params)) - p_prior(params) / targets.shape[0]


p_min, p_max = model.params - 10, model.params + 10
fig = contour_plot(jax.vmap(loss), min=p_min, max=p_max, res=100, iso=True, alpha=0.5, zorder=-1, label=r'\mathcal{L}(f,\theta)')

# fig = contour_plot(jax.vmap(loss), min=p_min, max=p_max, res=100, fig=None)
ax = fig.axes[0]
ax.scatter(*model.params)

# %%
fig = surf_plot(jax.vmap(loss), min=p_min, max=p_max, res=100)
ax = fig.axes[0]
ax.scatter(*jnp.append(model.params, loss(model.params)))

# %% [markdown]
# ## Laplace Approximation

# %%
from bayax.curv.ggn import ggn
from bayax.utils.helper import wrap_pytree_function
from bayax.operators.psd_operator import PSDOperator

ggn_fn = wrap_pytree_function(
    ggn(
        p=p_ll,
        f=model,
        X=samples,
        y=targets,
        scaling=float(samples.shape[0])
    ),
    nnx.state(model)
)
ggn_lr = PSDOperator(lambda v: ggn_fn(model.params, v), op_size=num_params).lowrank(num_modes=num_params)
cov_op = (ggn_lr + p_prior._cov.inverse()).inverse()
posterior = MultivariateNormal(cov=cov_op, mean=model.params)

# %%
from bayax.integrate import integrate, ef, ode23, ode45
dt, T = 0.01, 5.0
params_samples = posterior.sample(size=10)
x0 = jnp.expand_dims(model.params, axis=0).repeat(params_samples.shape[1], axis=0)
v0 = params_samples.T - x0

# %%
from bayax.approx.mc import pred_posterior_mean, pred_posterior_std, pred_posterior
mean_fn = pred_posterior_mean(model, params_samples)
std_fn = pred_posterior_std(model, params_samples)
pred_posterior_fn = pred_posterior(model, params_samples, p_ll)

# %%
from bayax.utils.plot import contour_plot, surf_plot
fig = contour_plot(mean_fn, min=[-3, -3], max=[3, 3], res=100, iso=False, alpha=0.5, zorder=-1, label=r'\mu[p(y|f)]')
ax = fig.axes[0]
ax.scatter(samples_c1[:, 0], samples_c1[:, 1], label='class 1', color='green', alpha=0.5)
ax.scatter(samples_c2[:, 0], samples_c2[:, 1], label='class 2', color='orange', alpha=0.5)
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_aspect('equal', 'box')
ax.legend()

# %%
from bayax.utils.plot import contour_plot, surf_plot
fig = contour_plot(std_fn, min=[-3, -3], max=[3, 3], res=100, iso=False, alpha=0.5, zorder=-1, label=r'\sigma[p(y|f)]')
ax = fig.axes[0]
ax.scatter(samples_c1[:, 0], samples_c1[:, 1], label='class 1', color='green', alpha=0.5)
ax.scatter(samples_c2[:, 0], samples_c2[:, 1], label='class 2', color='orange', alpha=0.5)
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_aspect('equal', 'box')
ax.legend()

# %%
from bayax.utils.plot import contour_plot, surf_plot
fig = contour_plot(std_fn, min=[-3, -3], max=[3, 3], res=100, iso=False, alpha=0.5, zorder=-1, label=r'\sigma[p(y|f)]')
ax = fig.axes[0]
ax.scatter(samples_c1[:, 0], samples_c1[:, 1], label='class 1', color='green', alpha=0.5)
ax.scatter(samples_c2[:, 0], samples_c2[:, 1], label='class 2', color='orange', alpha=0.5)
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_aspect('equal', 'box')
ax.legend()
