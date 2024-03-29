from typing import Union
from functools import cached_property

import jax.numpy as jnp
import jax
import equinox as eqx

from cryojax.simulator import ImagePipeline


def get_pointer_to_params(pipeline):
    output = (
        pipeline.specimen.pose.offset_x,
        pipeline.specimen.pose.offset_y,
        pipeline.specimen.pose.view_phi,
        pipeline.specimen.pose.view_theta,
        pipeline.specimen.pose.view_psi,
        pipeline.instrument.optics.ctf.defocus_u,
        pipeline.instrument.optics.ctf.defocus_v,
        pipeline.instrument.optics.ctf.amplitude_contrast,
    )
    return output


def get_pointer_to_density(pipeline):
    output = pipeline.specimen.density.fourier_density_grid
    return output


def update_pipeline(params: jax.Array, density, pipeline: ImagePipeline) -> ImagePipeline:
    new_pipeline = eqx.tree_at(get_pointer_to_density, pipeline, density)
    new_pipeline = eqx.tree_at(get_pointer_to_params, new_pipeline, params)

    return new_pipeline


@jax.jit
def compute_loss_(
    params: jax.Array, density: jax.Array, pipeline: ImagePipeline, observed: jax.Array
) -> jax.Array:
    pipeline = update_pipeline(params, density, pipeline)
    return jnp.sum((observed - pipeline.render(observed))**2)


@jax.jit
@jax.grad
def compute_grad(
    params: jax.Array, pipeline: ImagePipeline, observed: jax.Array
) -> jax.Array:
    return compute_loss_(params, pipeline, observed)


@jax.jit
def compute_loss(
    pipeline_weights: jax.Array,
    params: jax.Array,
    densities,
    pipelines: list[ImagePipeline],
    image_stack: jax.Array,
) -> jax.Array:
    comp_loss_map_pipelines_ = jax.vmap(compute_loss_, in_axes=(None, 0, None, None))
    comp_loss_map_images_ = jax.vmap(comp_loss_map_pipelines_, in_axes=(0, None, None, 0))

    lklhood_matrix = comp_loss_map_images_(
        params, densities, pipelines, image_stack
    )  # (N_pipelines, N_images)

    log_lklhood = jax.scipy.special.logsumexp(
        a=lklhood_matrix, b=pipeline_weights[None, :], axis=1
    )

    log_lklhood = jnp.sum(log_lklhood)

    return log_lklhood
