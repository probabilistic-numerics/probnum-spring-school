from typing import Callable, Tuple

import numpy as np
from probnum import backend


class Dataset:
    """Generate a dataset.

    Parameters
    ----------
    latent_fn
        Latent function.
    num
        Number of (training) data points.
    """

    def __init__(
        self,
        latent_fn: Callable[[backend.Array], backend.Array],
        num: int,
        data_shape: Tuple[int, ...] = (2,),
        seed: int = 0,
    ) -> None:
        self._latent_fn = latent_fn
        self._num = num
        self._data_shape = data_shape

        # Sample training data
        n_mixture_means = 5
        rng_state, rng_state_mixture_means = backend.random.split(
            backend.random.rng_state(seed)
        )
        mixture_means = 0.5 * backend.random.standard_normal(
            rng_state_mixture_means, shape=(n_mixture_means,) + data_shape
        ) + 0.5 * backend.ones(data_shape)
        X_list = []
        for mixture_mean in mixture_means:
            rng_state, rng_state_gaussian_comp = backend.random.split(rng_state)
            X_list.append(
                0.1
                * backend.random.standard_normal(
                    rng_state_gaussian_comp,
                    shape=(num // n_mixture_means,) + data_shape,
                )
                + mixture_mean
            )

        self._X = backend.vstack(X_list)
        out_of_bounds_mask = (self._X >= 1.0) | (self._X <= 0.0)
        rng_state, rng_state_oob = backend.random.split(rng_state)
        self._X[out_of_bounds_mask] = backend.random.uniform(
            rng_state_oob,
            shape=(backend.sum(out_of_bounds_mask),),
            minval=0.0,
            maxval=1.0,
        )

        # Noise on targets
        noise_scale = 300
        rng_state, rng_state_noise = backend.random.split(rng_state)
        self._y = latent_fn(self._X) + noise_scale * backend.random.standard_normal(
            rng_state_noise, shape=(num,)
        )

    @property
    def latent_fn(self) -> Callable[[backend.Array], backend.Array]:
        return self._latent_fn

    @property
    def X(self) -> backend.Array:
        return self._X

    @property
    def y(self) -> backend.Array:
        return self._y
