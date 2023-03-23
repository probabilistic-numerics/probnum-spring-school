import math
from typing import Callable

import numpy as np
from probnum import backend
from scipy.interpolate import make_interp_spline


def movement_cost(
    curve: backend.Array,
    landscape: Callable[[backend.Array], float],
    num_timesteps: int = 10**4,
) -> backend.Array:
    """Compute the cost of moving along a curve.

    Parameters
    ----------
    curve
        curve with n steps in d-dimensional space given by an n x d array.
    landscape
        Function defining the landscape.
    num_timesteps
        Number of time steps taken along the curve.

    Returns
    -------
    Dictionary with keys:
    - ``"total_cost"``
        Total cost of moving along the curve.
    - ``"cumulative_cost"``
        Cumulative cost of moving along the curve.
    - ``"timepoints"``
        Time points in the curve domain [0, 1], where points were selected.
    """
    # Compute the time parametrization of the curve
    curve = np.unique(curve, axis=0)
    steps = curve[1:, :] - curve[0:-1, :]
    if len(steps) == 0:
        return {
            "total_cost": 0.0,
            "cumulative_cost": backend.zeros((num_timesteps,)),
            "elevation": landscape(curve[0, :]) * backend.ones((num_timesteps + 1,)),
            "timepoints": backend.linspace(0.0, 1.0, num_timesteps + 1),
        }

    timepoints = np.cumsum(backend.linalg.vector_norm(steps, axis=1), axis=0)
    timepoints = np.insert(timepoints, 0, 0.0) / timepoints[-1]

    # Interpolate curve
    curve_interp = make_interp_spline(timepoints, curve, k=1)
    time = backend.linspace(0.0, 1.0, num_timesteps + 1)
    points_on_curve = curve_interp(time)

    # Length of the curve segments
    curve_segment_lengths = backend.linalg.vector_norm(
        points_on_curve[1:, :] - points_on_curve[0:-1, :], axis=1
    )

    # Elevation change along curve
    elevation_change = landscape(points_on_curve[1:, :]) - landscape(
        points_on_curve[0:-1, :]
    )

    # Compute cost of moving along curve
    uphill_mask = elevation_change >= 0.0
    downhill_mask = elevation_change < 0.0

    movement_cost_per_step = curve_segment_lengths
    movement_cost_per_step[uphill_mask] *= (
        1.0 + num_timesteps * elevation_change[uphill_mask]
    ) ** 2
    movement_cost_per_step[downhill_mask] *= backend.exp(
        num_timesteps * elevation_change[downhill_mask]
    )

    return {
        "total_cost": backend.sum(movement_cost_per_step),
        "cumulative_cost": np.cumsum(movement_cost_per_step),
        "elevation": landscape(points_on_curve),
        "timepoints": timepoints,
    }
