import os
import pickle
from matplotlib import transforms
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Roman"],
})

# plots = ['J', 'diff_Q', 'Q', 'residuals', 'test_datasets']
plots = ['Q', 'residuals', 'test_datasets']

log_dir = "logs"
n_task = 10
alpha_rect = np.linspace(.75, 0, 10)


def plot_performance(methods, labels, colors, type, type_label, n_exp, processing_fn=None, path=None,
                     fontsize=6, ticksize=6, topsize=4, figsize=(2.56, 1.5), limits=(0.205, 0.21, 0.76, 0.7),
                     top_x_off=-5, top_y_off=1):
    n_iters = list()
    add_n_iters = True
    f = plt.figure(figsize=figsize)
    ax = plt.Axes(f, limits)
    f.add_axes(ax)

    lines = list()
    for method, color in zip(methods, colors):
        data_path = os.path.join(log_dir, method, "%s.npy" % type)
        data_exp = np.load(data_path, allow_pickle=True)

        # Load the data
        data_list = list()
        for i in range(n_exp):
            data_list_exp = list()
            for j in range(n_task):
                cur_data = data_exp[i, j] if processing_fn is None else processing_fn(data_exp[i, j])
                data_list_exp += cur_data
                if add_n_iters:
                    n_iters.append(len(cur_data))

            data_list.append(data_list_exp)
            add_n_iters = False
        datas = np.array(data_list)

        mean_data = datas.mean(0)
        err_data = 2 * datas.std(0) / np.sqrt(datas.shape[0])
        lines.append(ax.plot(mean_data, linewidth=2, color=color)[0])
        ax.fill_between(np.arange(datas.shape[1]), mean_data - err_data, mean_data + err_data, alpha=.5, color=color)

    n_iters = np.append([0], np.cumsum(n_iters))

    ylim = ax.get_ylim()
    ax.text(15, ylim[1] + top_y_off, r"$\ldots$", va="bottom", fontsize=topsize)
    for i in range(len(n_iters) - 1):
        diff = n_iters[i + 1] - n_iters[i]
        rect = patches.Rectangle((n_iters[i], ylim[0]), diff, ylim[1] - ylim[0], color='black', alpha=alpha_rect[i])
        ax.axvline(n_iters[i], color='k', alpha=.75, linestyle='--', linewidth=1)
        ax.add_patch(rect)
        if i > 3:
            ax.text(n_iters[i] + 0.5 * diff + top_x_off, ylim[1] + top_y_off,
                    r"$\mathcal{T}_{%d}$" % (i + 1), va="bottom", fontsize=topsize)
    ax.set_ylim(ylim)
    ax.set_xlim([0, n_iters[-1]])

    ax.grid()
    ax.set_xlabel("Iteration", fontsize=fontsize)
    ax.set_ylabel(type_label, fontsize=fontsize)

    ax.set_xticks(np.append(np.linspace(0, 250, 6), [mean_data.size]))
    # ax.set_yticks(fontsize=ticksize)
    ax.legend(lines, labels, ncol=2, fontsize=ticksize)
    if type == 'J':
        ax.plot(np.ones(datas.shape[1]) * -47.557563488, '--', color='k', linewidth=1)

    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.tick_params(axis='both', which='minor', labelsize=ticksize)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def plot_q_values(methods, labels, n_exp, figsize=(1.6, 1.5), fontsize=9, ticksize=6, path=None):
    f = plt.figure(figsize=figsize)

    for i, (method, label) in enumerate(zip(methods, labels)):
        ax = f.add_axes([0.07 + (i % 2) * 0.4, 0.075 + (1 - (i // 2)) * 0.465, 0.4, 0.375])
        # ax.axis('off')
        datas = np.load(os.path.join(log_dir, method, "Q.npy"), allow_pickle=True)

        Qs = np.zeros(((n_exp,) + datas[0, 0][0].shape))
        for k in range(len(Qs)):
            Qs[k] = datas[k, -1][-1]
        mean_Qs = Qs.mean(0).max(1).reshape(21, 21)

        ax.set_title(label, fontsize=0.8 * fontsize, pad=2)
        im = ax.imshow(mean_Qs.T, origin='lower', cmap='inferno', vmin=mean_Qs.min(), vmax=mean_Qs.max())
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticks([])
        ax.set_yticks([])

        # This is the case if the plot is on the left edge -> ylabels
        if i % 2 == 0:
            ax.set_ylabel(r"$y$", fontsize=fontsize)

        # This is the case if the plot is on the bottom -> xlabels
        if (i // 2) == 1:
            ax.set_xlabel(r"$x$", fontsize=fontsize, labelpad=1.)

    cbar_ax = f.add_axes([0.9, 0.1, 0.07, 0.8])
    cbar_ax.tick_params(axis='both', which='major', labelsize=ticksize)
    cbar_ax.tick_params(axis='both', which='minor', labelsize=ticksize)
    cb = plt.colorbar(im, cax=cbar_ax)
    cb.set_ticks([])
    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def visualize_paths(methods, labels, colors, figsize=(2., 1.5), fontsize=9, path=None):
    f = plt.figure(figsize=figsize)
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        ax = f.add_axes([0. + (i % 2) * 0.51, 0.075 + (1 - (i // 2)) * 0.465, 0.49, 0.375])

        pre_processed_path = os.path.join(log_dir, method, "processed_test_datasets.pkl")
        if os.path.exists(pre_processed_path):
            with open(pre_processed_path, "rb") as file:
                xy, diff = pickle.load(file)
        else:
            data_path = os.path.join(log_dir, method, "test_datasets.pkl")

            with open(data_path, 'rb') as file:
                dataset = pickle.load(file)
            trajectories = [d[-1][-1] for d in dataset]
            xy = [np.array([tr[0] for tr in traj]) for traj in trajectories]
            diff = [np.array([tr[3] - tr[0] for tr in traj]) for traj in trajectories]

            with open(pre_processed_path, "wb") as file:
                pickle.dump((xy, diff), file)

        ax.set_title(label, fontsize=0.8 * fontsize, pad=2)
        for i in range(len(xy)):
            start = patches.Rectangle((0, 0), .02, .02, color='red')
            goal = patches.Rectangle((.89, .9), .02, .02, linewidth=0.5, color='black')
            wall0 = patches.Rectangle((0, 0), 1, 1, linewidth=0.5, edgecolor='k', facecolor=(0.8, 0.8, 0.8))
            wall1 = patches.Rectangle((.15, 0), 0.1, .95, linewidth=0.5, edgecolor='k', facecolor='grey',
                                      hatch='/')
            wall2 = patches.Rectangle((.55, 0.05), 0.1, .95, linewidth=0.5, edgecolor='k', facecolor='grey',
                                      hatch='/')
            ax.add_patch(wall0)
            ax.add_patch(start)
            ax.add_patch(goal)
            ax.add_patch(wall1)
            ax.add_patch(wall2)

            xy_exp = xy[i]
            diff_exp = diff[i] / 0.05
            # > Non-dimensionalization
            ax.quiver(xy_exp[:, 0], xy_exp[:, 1], diff_exp[:, 0], diff_exp[:, 1],
                      units='xy', width=0.01, alpha=.5, scale=20., zorder=10, color=color)

            ax.set_xlim(0., 1.)
            ax.set_ylim(0., 1.)
            ax.axis('off')

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def visualize_environments(figsize=(1.3, 1.5), path=None):
    f = plt.figure(figsize=figsize)
    wall_sizes = [0.05, 0.35, 0.65, 0.95]
    for i in range(0, 4):
        ax = f.add_axes([0.005 + (i % 2) * 0.51, 0.02 + (1 - (i // 2)) * 0.46, 0.48, 0.48])

        start = patches.Rectangle((0, 0), .03, .03, color='red')
        goal = patches.Rectangle((.89, .9), .03, .03, linewidth=0.5, color='black')
        wall0 = patches.Rectangle((0, 0), 1, 1, linewidth=0.5, edgecolor='k', facecolor=(0.8, 0.8, 0.8))
        wall1 = patches.Rectangle((.15, 0), 0.1, wall_sizes[i], linewidth=0.5, edgecolor='k', facecolor='grey',
                                  hatch='/')
        wall2 = patches.Rectangle((.55, 1 - wall_sizes[i]), 0.1, wall_sizes[i], linewidth=0.5, edgecolor='k',
                                  facecolor='grey', hatch='/')
        ax.add_patch(wall0)
        ax.add_patch(start)
        ax.add_patch(goal)
        ax.add_patch(wall1)
        ax.add_patch(wall2)
        ax.set_xlim(0., 1.)
        ax.set_ylim(0., 1.)
        ax.axis("equal")
        ax.axis('off')

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    n_exps = 40
    plot_performance(
        ["boosted_curriculum", "boosted_no_curriculum", "no_boosted_curriculum", "no_boosted_no_curriculum"],
        ["BC-FQI", "B-FQI", "C-FQI", "FQI"], ["C0", "C1", "C2", "C3"], "J", "Cum. Disc. Return", n_exps, fontsize=9,
        ticksize=6, topsize=6, figsize=(2.56, 1.5), limits=[0.205, 0.21, 0.757, 0.71], top_y_off=1.1,
        path="figures/puddle_world_performance.pdf")
    plot_performance(
        ["boosted_curriculum", "boosted_no_curriculum", "no_boosted_curriculum", "no_boosted_no_curriculum"],
        ["BC-FQI", "B-FQI", "C-FQI", "FQI"], ["C0", "C1", "C2", "C3"], "diff_Q", r"$\| Q_t^k - Q_t^* \|_{1, \mu}$",
        n_exps, fontsize=9, ticksize=6, topsize=6, limits=[0.16, 0.21, 0.805, 0.71], figsize=(2.56, 1.5),
        top_y_off=1.1, path="figures/puddle_world_diff_q.pdf")
    plot_q_values(["boosted_curriculum", "boosted_no_curriculum", "no_boosted_curriculum", "no_boosted_no_curriculum"],
                  ["BC-FQI", "B-FQI", "C-FQI", "FQI"], n_exps, path="figures/puddle_world_q.pdf")
    plot_performance(["boosted_curriculum"], ["BC-FQI"], ["C0"], "residuals",
                     r'$\dfrac{\sum_a \varrho_t(s_0,a)}{\#\mathcal{A}}$', n_exps,
                     processing_fn=lambda residuals: [np.array(res).mean(-1)[..., 0] for res in residuals],
                     figsize=(3.2, 1.5), topsize=6, fontsize=9, top_y_off=0.5, limits=[0.2, 0.216, 0.77, 0.7],
                     path="figures/puddle_world_residuals.pdf")
    visualize_paths(
        ["boosted_curriculum", "boosted_no_curriculum", "no_boosted_curriculum", "no_boosted_no_curriculum"],
        ["BC-FQI", "B-FQI", "C-FQI", "FQI"], ["C0", "C1", "C2", "C3"], path="figures/puddle_world_paths.pdf")
    visualize_environments(path="figures/puddle_world_env.pdf")
