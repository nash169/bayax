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
from flax import nnx
from jax import grad, jit, vmap, random
import optax
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join('../')))
from bayax.models import MLP
from bayax.utils.train import DataLoader, OptCfg, TrainCfg, train
from bayax.utils.helper import make_sinusoid
from bayax.utils.plot import plot, scatter

# %%
sinus_factor = 5.0
X_train, y_train, _, _, X_test, y_test = make_sinusoid(
    n_train=150, n_valid=50, n_test=100,
    interval=(-2.0, 2.0),
    noise=0.15,
    sinus_factor=sinus_factor
)

# %%
model = MLP(
    layers=[1, 8, 8, 1],
    nl=nnx.tanh,
    use_bias=True,
)
num_params = model.size

# %%
fig = scatter(jnp.vstack((X_train, y_train)).T, color="blue", alpha=0.6, label="Training data")
fig = scatter(jnp.vstack((X_test, y_test)).T, fig=fig, color="green", alpha=0.6, label="Test data")
fig = plot(lambda x: jnp.sin(sinus_factor * x), fig=fig, range=[-2, 2], color="black", linestyle="--", label="True function")
fig = plot(model, fig=fig, range=[-2, 2], color="red", label="Prediction")

# %%
from bayax.densities import Normal, MultivariateNormal
from bayax.functions.likelihood import neg_logll
from bayax.operators import DiagOperator


likelihood = lambda f: Normal(var=jnp.array(1.), mean=f)

prior_cov = DiagOperator(
    diag=jnp.array(10.),
    dim=num_params
)
prior = MultivariateNormal(cov=prior_cov)


# %%
def loss_fn(model, x, y):
    y_pred = model(x)
    return neg_logll(likelihood, y, y_pred)


cfg = TrainCfg(
    opt=optax.adam(1e-1),
    steps=1000,
    batch_size=X_train.shape[0],
    verbose=True
)
loss_val = train(model, X_train.reshape(-1, 1), y_train, loss_fn, cfg)

# %%
fig = scatter(jnp.vstack((X_train, y_train)).T, color="blue", alpha=0.6, label="Training data")
fig = scatter(jnp.vstack((X_test, y_test)).T, fig=fig, color="green", alpha=0.6, label="Test data")
fig = plot(lambda x: jnp.sin(sinus_factor * x), fig=fig, range=[-2, 2], color="black", linestyle="--", label="True function")
fig = plot(model, fig=fig, range=[-2, 2], color="red", label="Prediction")
