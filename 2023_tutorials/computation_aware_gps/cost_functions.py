import math
from typing import Callable

import numpy as np
from probnum import backend
from scipy.interpolate import make_interp_spline


def movement_cost(
    curve: backend.Array,
    landscape: Callable[[backend.Array], float],
    num_steps: int = 10**3,
) -> backend.Array:
    """Compute the cost of moving along a curve.

    Parameters
    ----------
    curve
        curve with n steps in d-dimensional space given by an n x d array.
    landscape
        Function defining the landscape.
    num_steps
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
    arc_lengths_to_points = np.insert(
        np.cumsum(np.linalg.norm(steps, axis=1), axis=0), 0, 0.0
    )
    arc_length = arc_lengths_to_points[-1]
    arc_lengths = backend.linspace(0.0, arc_length, num_steps + 1)

    if len(steps) == 0:
        # TODO
        return {
            "total_cost": 0.0,
            "arc_length": arc_length,
            "cumulative_cost": backend.zeros((num_steps + 1,)),
            "arc_lengths": arc_lengths,
            "arc_lengths_to_points": arc_lengths_to_points,
            "elevation": landscape(curve[0, :]) * backend.ones((num_steps + 1,)),
        }

    # Interpolate curve
    curve_interp = make_interp_spline(arc_lengths_to_points, curve, k=1)
    interpolated_points_on_curve = curve_interp(arc_lengths)

    # Gradient along curve
    gradient = (
        (
            landscape(interpolated_points_on_curve[1:, :])
            - landscape(interpolated_points_on_curve[0:-1, :])
        )
        / arc_length
        * num_steps
    )

    # Compute cost of moving along curve
    uphill_mask = gradient >= 0.0
    downhill_mask = gradient < 0.0

    movement_cost_per_step = arc_lengths[1:] - arc_lengths[0:-1]
    movement_cost_per_step[uphill_mask] *= (1.0 + gradient[uphill_mask]) ** 2
    movement_cost_per_step[downhill_mask] *= backend.exp(gradient[downhill_mask])

    return {
        "total_cost": backend.sum(movement_cost_per_step),
        "arc_length": arc_length,
        "cumulative_cost": np.insert(np.cumsum(movement_cost_per_step), 0, 0.0),
        "arc_lengths": arc_lengths,
        "arc_lengths_to_points": arc_lengths_to_points,
        "elevation": landscape(interpolated_points_on_curve),
        "gradient": gradient,
    }
