
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence
from dynamax.generalized_gaussian_ssm.models import ParamsGGSSM
from dynamax.generalized_gaussian_ssm.inference import conditional_moments_gaussian_filter, EKFIntegrals
from flax import nnx
import jax.random as jr
from jax.flatten_util import ravel_pytree
from functools import partial

class CNN(nnx.Module):
  """A simple CNN model."""

  def __init__(self, *, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(1, 2, kernel_size=(3, 3), rngs=rngs)
    self.conv2 = nnx.Conv(2, 2, kernel_size=(3, 3), rngs=rngs)
    self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.linear1 = nnx.Linear(98, 4, rngs=rngs)
    self.linear2 = nnx.Linear(4, 1, rngs=rngs)

  def __call__(self, x):
    x = self.avg_pool(nnx.relu(self.conv1(x)))
    x = self.avg_pool(nnx.relu(self.conv2(x)))
    x = x.reshape(-1)  # flatten
    x = nnx.relu(self.linear1(x))
    x = self.linear2(x)
    return x

class CNN_EKF_Training:
    def __init__(self, initial_sigma2=0.1, dynamic_sigma=1e-4):
        self.initial_sigma2 = initial_sigma2
        self.dynamic_sigma = dynamic_sigma

    def get_cnn_flattened_params(self, key=0):
        model = CNN(rngs=nnx.Rngs(0))
        graphdef, state = nnx.split(model)
        flat_params, unflatten_fn = ravel_pytree(state)

        def apply(flat_params, x, model, graphdef, unflatten_fn):
            model = nnx.merge(graphdef, unflatten_fn(flat_params))
            return model(x)

        apply_fn = partial(apply, model=model, graphdef=graphdef, unflatten_fn=unflatten_fn)
        return model, graphdef, flat_params, unflatten_fn, apply_fn

    def run_cnn_ekf(self, X_train, y_train, sample_size):
        model, graphdef, flat_params, unflatten_fn, apply_fn = self.get_cnn_flattened_params()

        state_dim = flat_params.size
        softmax_fn = lambda w, x: jax.nn.sigmoid(apply_fn(w, x))

        initial_mean = flat_params
        initial_covariance = jnp.eye(state_dim) * self.initial_sigma2
        dynamics_function = lambda w, x: w
        dynamics_covariance = jnp.eye(state_dim) * self.dynamic_sigma
        emission_mean_function = lambda w, x: softmax_fn(w, x)
        emission_cov_function = lambda w, x: softmax_fn(w, x) * (1 - softmax_fn(w, x))

        cmgf_ekf_params = ParamsGGSSM(
            initial_mean=initial_mean,
            initial_covariance=initial_covariance,
            dynamics_function=dynamics_function,
            dynamics_covariance=dynamics_covariance,
            emission_mean_function=emission_mean_function,
            emission_cov_function=emission_cov_function
        )

        inputs = X_train[:sample_size]
        output = y_train[:sample_size].astype(jnp.float32)

        cmgf_ekf_post = conditional_moments_gaussian_filter(cmgf_ekf_params, EKFIntegrals(), output, inputs=inputs)

        w_means, w_covs = cmgf_ekf_post.filtered_means, cmgf_ekf_post.filtered_covariances
        post_mean, post_cov = w_means[-1], w_covs[-1]

        model_trained = nnx.merge(graphdef, unflatten_fn(post_mean))

        return w_means, w_covs, model_trained
