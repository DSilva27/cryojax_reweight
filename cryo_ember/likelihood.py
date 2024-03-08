from typing import Union
from functools import cached_property

import jax.numpy as jnp
import jax
import equinox as eqx
import cryojax.simulator as cs

from cryojax.simulator import ImagePipeline


def get_pointer_to_params(pipeline):
    if not isinstance(pipeline.specimen.pose, cs.EulerAnglePose):
        raise ValueError("Only a `EulerAnglePose` pose representation is supported.")
    output = (
        pipeline.specimen.pose.offset_x_in_angstroms,
        pipeline.specimen.pose.offset_y_in_angstroms,
        pipeline.specimen.pose.view_phi,
        pipeline.specimen.pose.view_theta,
        pipeline.specimen.pose.view_psi,
        pipeline.instrument.optics.ctf.defocus_u_in_angstroms,
        pipeline.instrument.optics.ctf.defocus_v_in_angstroms,
        pipeline.instrument.optics.ctf.amplitude_contrast_ratio,
    )
    return output


def get_pointer_to_density(pipeline):
    if not isinstance(pipeline.specimen.pose, cs.FourierVoxelGrid):
        raise ValueError("Only a `FourierVoxelGrid` scattering potential representation is supported.")
    output = pipeline.specimen.potential.fourier_voxel_grid
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
