import jax.numpy as jnp
import jax
import equinox as eqx

from cryojax.simulator import FourierVoxelGrid
from cryojax.inference.distributions import AbstractDistribution


def get_pointer_to_params(distribution: AbstractDistribution):
    output = (
        distribution.pipeline.specimen.pose.offset_x,
        distribution.pipeline.specimen.pose.offset_y,
        distribution.pipeline.specimen.pose.view_phi,
        distribution.pipeline.specimen.pose.view_theta,
        distribution.pipeline.specimen.pose.view_psi,
        distribution.pipeline.instrument.optics.ctf.defocus_u,
        distribution.pipeline.instrument.optics.ctf.defocus_v,
        distribution.pipeline.instrument.optics.ctf.amplitude_contrast,
    )
    return output


def get_pointer_to_density(distribution: AbstractDistribution):
    if not isinstance(distribution.pipeline.specimen.density, FourierVoxelGrid):
        raise AttributeError("")
    output = distribution.pipeline.specimen.density.fourier_density_grid
    return output


def update_distribution(
    params: jax.Array, density, distribution: AbstractDistribution
) -> AbstractDistribution:
    new_distribution = eqx.tree_at(get_pointer_to_density, distribution, density)
    new_distribution = eqx.tree_at(get_pointer_to_params, new_distribution, params)

    return new_distribution


@jax.jit
def compute_loss_(
    params: jax.Array,
    density: jax.Array,
    distribution: AbstractDistribution,
    observed: jax.Array,
) -> jax.Array:
    distribution = update_distribution(params, density, distribution)
    return distribution.log_probability(observed)


@jax.jit
@jax.grad
def compute_grad(
    params: jax.Array, distribution: AbstractDistribution, observed: jax.Array
) -> jax.Array:
    return compute_loss_(params, distribution, observed)


@jax.jit
def compute_loss(
    distribution_weights: jax.Array,
    params: jax.Array,
    densities,
    distributions: AbstractDistribution,
    image_stack: jax.Array,
) -> jax.Array:
    comp_loss_map_distributions_ = jax.vmap(
        compute_loss_, in_axes=(None, 0, None, None)
    )
    comp_loss_map_images_ = jax.vmap(
        comp_loss_map_distributions_, in_axes=(0, None, None, 0)
    )

    lklhood_matrix = comp_loss_map_images_(
        params, densities, distributions, image_stack
    )  # (N_distributions, N_images)

    log_lklhood = jax.scipy.special.logsumexp(
        a=lklhood_matrix, b=distribution_weights[None, :], axis=1
    )

    log_lklhood = jnp.sum(log_lklhood)

    return log_lklhood
