import matplotlib.pyplot as plt
import numpy as np


def plot_path_cost(path_cost1, path_cost2):
    fig, axs = plt.subplots(
        nrows=3, ncols=2, sharex="col", sharey="row", height_ratios=(1.0, 0.5, 0.5)
    )

    for idx, mc in enumerate([path_cost1, path_cost2]):
        # Energy cost
        axs[0, idx].plot(
            mc["arc_lengths"],
            mc["cumulative_cost"],
            color=["C3", "C1"][idx],
            label=f"Energy cost of path {idx+1}: {mc['total_cost']:.4f}",
        )
        axs[0, idx].legend(loc="upper center")

        # Curve points
        axs[0, idx].vlines(
            mc["arc_lengths_to_points"],
            ymin=0.0,
            ymax=np.maximum(path_cost1["total_cost"], path_cost2["total_cost"]),
            color="black",
            alpha=0.3,
            zorder=-10,
        )

        # Gradient
        axs[1, idx].plot(
            mc["arc_lengths"][1:],
            mc["gradient"],
            color="black",
        )
        axs[1, idx].axhline(color="black", alpha=0.2, linestyle="--")

        # Altitude
        axs[2, idx].fill_between(
            mc["arc_lengths"],
            mc["elevation"],
            np.min(np.concatenate([path_cost1["elevation"], path_cost2["elevation"]])),
            color="black",
            alpha=0.2,
        )
        axs[2, idx].set(xlabel="Arc Length")

    axs[0, 0].set(ylabel="Energy cost")
    axs[1, 0].set(ylabel="Gradient")
    axs[2, 0].set(ylabel="Altitude")

    fig.align_ylabels()

    plt.show()
