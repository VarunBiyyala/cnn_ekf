
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
from functools import partial
from typing import Sequence
from jax.flatten_util import ravel_pytree
from dynamax.generalized_gaussian_ssm.models import ParamsGGSSM
from dynamax.generalized_gaussian_ssm.inference import conditional_moments_gaussian_filter, EKFIntegrals

class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

class MLP_EKF_Training:
    def __init__(self, model_dims=[2, 16, 16, 1], initial_sigma2=0.1, dynamic_sigma=1e-4):
        self.model_dims = model_dims
        self.initial_sigma2 = initial_sigma2
        self.dynamic_sigma = dynamic_sigma

    def get_mlp_flattened_params(self, model_dims, key=0):
        if isinstance(key, int):
            key = jr.PRNGKey(key)

        input_dim, features = model_dims[0], model_dims[1:]
        model = MLP(features)
        dummy_input = jnp.ones((input_dim,))

        params = model.init(key, dummy_input)
        flat_params, unflatten_fn = ravel_pytree(params)

        def apply(flat_params, x, model, unflatten_fn):
            return model.apply(unflatten_fn(flat_params), jnp.atleast_1d(x))

        apply_fn = partial(apply, model=model, unflatten_fn=unflatten_fn)

        return model, flat_params, unflatten_fn, apply_fn

    def run_mlp_ekf(self, output, inputs):
        input_dim, hidden_dims, output_dim = 2, [16, 16], 1
        model_dims = [input_dim, *hidden_dims, output_dim]

        model, flat_params, unflatten_fn, apply_fn = self.get_mlp_flattened_params(model_dims)

        state_dim, emission_dim = flat_params.size, output_dim
        sigmoid_fn = lambda w, x: jax.nn.sigmoid(apply_fn(w, x))

        initial_mean = flat_params
        initial_covariance = jnp.eye(state_dim) * self.initial_sigma2
        dynamics_function = lambda w, x: w
        dynamics_covariance = jnp.eye(state_dim) * self.dynamic_sigma
        emission_mean_function = lambda w, x: sigmoid_fn(w, x)
        emission_cov_function = lambda w, x: sigmoid_fn(w, x) * (1 - sigmoid_fn(w, x))

        cmgf_ekf_params = ParamsGGSSM(
            initial_mean=initial_mean,
            initial_covariance=initial_covariance,
            dynamics_function=dynamics_function,
            dynamics_covariance=dynamics_covariance,
            emission_mean_function=emission_mean_function,
            emission_cov_function=emission_cov_function
        )

        cmgf_ekf_post = conditional_moments_gaussian_filter(cmgf_ekf_params, EKFIntegrals(), output, inputs=inputs)

        return cmgf_ekf_post.filtered_means, cmgf_ekf_post.filtered_covariances, sigmoid_fn

