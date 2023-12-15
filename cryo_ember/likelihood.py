"""
Image formation models, equipped with log-likelihood functions.
"""

from __future__ import annotations

__all__ = ["GaussianImage"]

from typing import Union
from functools import cached_property

import jax.numpy as jnp
import jax
import equinox as eqx

from cryojax.simulator.likelihood import GaussianImage


def get_pointer_to_params(model):
    output = (
        model.specimen.pose.offset_x,
        model.specimen.pose.offset_y,
        model.specimen.pose.view_phi,
        model.specimen.pose.view_theta,
        model.specimen.pose.view_psi,
        model.instrument.optics.defocus_u,
        model.instrument.optics.defocus_v,
        model.instrument.optics.amplitude_contrast,
    )
    return output


def get_pointer_to_density(model):
    output = model.specimen.density.weights
    return output


def update_model(params: jax.Array, density, model: GaussianImage) -> GaussianImage:
    new_model = eqx.tree_at(get_pointer_to_density, model, density)
    new_model = eqx.tree_at(get_pointer_to_params, new_model, params)

    return new_model


@jax.jit
def compute_loss_(
    params: jax.Array, density: jax.Array, model: GaussianImage, observed: jax.Array
) -> jax.Array:
    model = update_model(params, density, model)
    return model(observed)


@jax.jit
@jax.grad
def compute_grad(
    params: jax.Array, model: GaussianImage, observed: jax.Array
) -> jax.Array:
    return compute_loss_(params, model, observed)


@jax.jit
def compute_loss(
    model_weights: jax.Array,
    params: jax.Array,
    densities,
    models: list[GaussianImage],
    image_stack: jax.Array,
) -> jax.Array:
    comp_loss_map_models_ = jax.vmap(compute_loss_, in_axes=(None, 0, None, None))
    comp_loss_map_images_ = jax.vmap(comp_loss_map_models_, in_axes=(0, None, None, 0))

    lklhood_matrix = comp_loss_map_images_(
        params, densities, models, image_stack
    )  # (N_models, N_images)

    log_lklhood = jax.scipy.special.logsumexp(
        a=lklhood_matrix, b=model_weights[None, :], axis=1
    )

    log_lklhood = jnp.sum(log_lklhood)

    return log_lklhood
