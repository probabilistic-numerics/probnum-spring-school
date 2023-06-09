from typing import Callable

from probnum import backend
from probnum.randprocs import kernels


def generate_landscape(
    seed: int, x0_lims=[0.0, 1.0], x1_lims=[0.0, 1.0]
) -> Callable[[backend.Array], backend.Array]:
    """Generate a random landscape.

    Parameters
    ----------
    seed
        Seed for the landscape.
    """
    rng_state = backend.random.rng_state(seed)
    input_shape = (2,)

    # Large mountains and valleys
    n0 = 10
    kernel0 = kernels.Matern(input_shape, lengthscale=0.2, nu=3.5)
    rng_state, rng_state_rweights0, rng_state_X0 = backend.random.split(
        rng_state=rng_state, num=3
    )
    rweights0 = backend.random.uniform(
        rng_state_rweights0,
        shape=(n0,),
        minval=-1.0,
        maxval=1.0,
    )
    X0 = backend.random.uniform(
        rng_state_X0,
        shape=(n0, 2),
        minval=x0_lims[0],
        maxval=x0_lims[1],
    )

    # Small structures
    n1 = 40
    kernel1 = kernels.Matern(input_shape, lengthscale=0.1)
    rng_state, rng_state_rweights1, rng_state_X1 = backend.random.split(
        rng_state=rng_state, num=3
    )
    rweights1 = backend.random.uniform(
        rng_state_rweights1,
        shape=(n1,),
        minval=-0.3,
        maxval=0.3,
    )
    X1 = backend.random.uniform(
        rng_state_X1,
        shape=(n1, 2),
        minval=x1_lims[0],
        maxval=x1_lims[1],
    )

    # Complete landscape
    return (
        lambda x: kernel0.matrix(x, X0) @ rweights0 + kernel1.matrix(x, X1) @ rweights1
    )
