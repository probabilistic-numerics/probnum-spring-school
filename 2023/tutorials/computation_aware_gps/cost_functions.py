from typing import Callable

import numpy as np
from probnum import backend
from scipy.interpolate import make_interp_spline


def movement_cost(
    curve: backend.Array,
    landscape: Callable[[backend.Array], float],
    num_curve_steps: int = 10**4,
) -> backend.Array:
    """Compute the cost of moving along a curve.

    Parameters
    ----------
    curve
        curve with n steps in d-dimensional space given by an n x d array.
    landscape
        Function defining the landscape.
    num_curve_steps
        Number of steps taken along the curve.
    """
    # Compute the parametrization of the curve
    curve = np.unique(curve, axis=0)
    steps = curve[1:, :] - curve[0:-1, :]
    if len(steps) == 0:
        return 0.0

    curve_parametrization = np.cumsum(backend.linalg.vector_norm(steps, axis=1), axis=0)
    curve_parametrization = (
        np.insert(curve_parametrization, 0, 0.0) / curve_parametrization[-1]
    )

    # Interpolate curve
    curve_interp = make_interp_spline(curve_parametrization, curve, k=1)

    # Evaluate elevation change along curve
    distance_on_curve = backend.linspace(0.0, 1.0, num_curve_steps + 1)
    points_on_curve = curve_interp(distance_on_curve)
    elevation_change = landscape(points_on_curve[1:, :]) - landscape(
        points_on_curve[0:-1, :]
    )

    # Compute cost of moving along curve
    uphill_mask = elevation_change >= 0.0
    downhill_mask = elevation_change < 0.0
    cost_uphill_movement = backend.sum(
        np.exp(elevation_change[uphill_mask]) / num_curve_steps
    )
    cost_downhill_movement = backend.sum(
        np.exp(elevation_change[downhill_mask]) / num_curve_steps
    )

    return cost_uphill_movement + cost_downhill_movement
