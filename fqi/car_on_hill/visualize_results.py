import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Roman"],
})


def plot_lines(ax, data, color):
    # The data is of shape (n, y, z), where the n is the number of samples, y the number of envs and the z the
    # iterations in the environment
    reshaped_data = np.concatenate([data[:, i, :] for i in range(0, data.shape[1])], axis=-1)
    mean = np.mean(reshaped_data, axis=0)
    n = mean.shape[0]

    ax.plot(np.linspace(1, n, n), reshaped_data.T, color=color, alpha=0.5, linestyle="--", linewidth=0.5)
    l1, = ax.plot(np.linspace(1, n, n), mean, color=color, linewidth=2)

    return l1


def visualize_evolution(path=None, residuals=False, fontsize=6, ticksize=6, axsize=(0.17, 0.215, 0.825, 0.7),
                        yoffset=0.035):
    fig = plt.figure(figsize=(2.56, 1.5))
    ax = fig.add_axes(axsize)

    filename = "Q.npy" if residuals else "J.npy"

    boosted_curriculum_data = np.load(os.path.join("logs/boosted_curriculum", filename))
    boosted_data = np.load(os.path.join("logs/boosted_no_curriculum", filename))
    curriculum_data = np.load(os.path.join("logs/no_boosted_curriculum", filename))
    default_data = np.load(os.path.join("logs/no_boosted_no_curriculum", filename))

    l1 = plot_lines(ax, boosted_curriculum_data, "C0")
    l2 = plot_lines(ax, boosted_data, "C1")
    l3 = plot_lines(ax, curriculum_data, "C2")
    l4 = plot_lines(ax, default_data, "C3")

    ylim = ax.get_ylim()
    plt.vlines([20, 40], *ylim, color="black", linestyle="--", linewidths=[2, 2], alpha=0.5)
    patch = Rectangle([0, ylim[0]], 20, ylim[1] - ylim[0], color="black", alpha=0.4)
    fig.gca().add_patch(patch)
    patch = Rectangle([20, ylim[0]], 20, ylim[1] - ylim[0], color="black", alpha=0.2)
    fig.gca().add_patch(patch)

    plt.text(10, ylim[1] + yoffset, r"$\mathcal{T}_1$", fontsize=ticksize)
    plt.text(30, ylim[1] + yoffset, r"$\mathcal{T}_2$", fontsize=ticksize)
    plt.text(50, ylim[1] + yoffset, r"$\mathcal{T}_3$", fontsize=ticksize)

    plt.legend([l1, l2, l3, l4], ["BC-FQI", "B-FQI", "C-FQI", "FQI"], fontsize=ticksize, ncol=2)

    plt.xlabel("Iteration", fontsize=fontsize)
    plt.ylabel(r"$\| Q_t^k - Q_t^* \|_{1, \mu}$" if residuals else "Cum. Disc. Return", fontsize=fontsize)
    plt.gca().tick_params(axis='both', which='major', labelsize=ticksize)
    plt.xlim([0, 60])
    plt.ylim(ylim)

    plt.grid()
    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    visualize_evolution(path="figures/car_on_hill_performance.pdf", residuals=False, fontsize=9,
                        axsize=(0.195, 0.215, 0.78, 0.7), yoffset=0.035)
    visualize_evolution(path="figures/car_on_hill_diff_q.pdf", residuals=True, fontsize=9, axsize=(0.17, 0.215, 0.805, 0.7),
                        yoffset=0.02)
