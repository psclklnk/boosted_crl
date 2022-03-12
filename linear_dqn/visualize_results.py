import os
import pickle
import numpy as np
from mushroom_rl.core import Core
from linear_dqn.train_lspi_linear import LQRWeights, EpsGreedyPolicy, rollout_policy
from linear_dqn.dqn import BoostedDQN
from linear_dqn.envs import LinearSystem, NonLinearSystem
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Roman"],
})


def visualize_value_fn(ax, actions, approximator, features=None):
    X, Y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    xy = np.stack((X, np.zeros_like(X), Y, np.zeros_like(Y)), axis=-1)
    xy_flat = np.reshape(xy, (-1, 4))

    if features is None:
        max_q_flat = np.max(approximator(xy_flat), axis=1)
        max_id_q_flat = np.argmax(approximator(xy_flat), axis=1)
    else:
        full = np.concatenate((np.repeat(xy_flat[None, ...], 8, axis=0),
                               np.repeat(actions[:, None, ...], 2500, axis=1)), axis=-1)
        features_flat = features(np.reshape(full, (8 * 2500, -1)))
        max_q_flat = np.max(np.reshape(np.squeeze(approximator(features_flat)), (8, 2500)), axis=0)
        max_id_q_flat = np.argmax(np.reshape(np.squeeze(approximator(features_flat)), (8, 2500)), axis=0)

    Z = np.reshape(max_q_flat, (50, 50))
    max_id_q = np.reshape(max_id_q_flat, (50, 50))

    actions_x = actions[max_id_q, 0] / np.linalg.norm(actions[max_id_q, :], axis=-1)
    actions_y = actions[max_id_q, 1] / np.linalg.norm(actions[max_id_q, :], axis=-1)

    plt.quiver(X[0::4, 0::4], Y[0::4, 0::4], actions_x[0::4, 0::4], actions_y[0::4, 0::4], width=0.01)

    im = ax.imshow(Z, extent=[-5, 5, -5, 5], vmin=np.min(Z), vmax=np.max(Z))
    ax.scatter(0, 0, color="red")

    return im


def plot_dqn_policies_linear(seed):
    mdp = LinearSystem(discrete_actions=True, friction=0.1, full_init=False)
    agent = BoostedDQN.load(os.path.join("dqn_agents_1000", "seed-%d.pkl" % seed))
    core = Core(agent, mdp)

    dataset = core.evaluate(n_episodes=5)

    plt.scatter([d[0][0] for d in dataset], [d[0][2] for d in dataset])
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.text(-4, 4, "Trained Agent In Dist")
    plt.show()

    eval_mdp = LinearSystem(discrete_actions=True, friction=0.1, full_init=True)
    core = Core(agent, eval_mdp)
    dataset = core.evaluate(n_episodes=5)

    plt.scatter([d[0][0] for d in dataset], [d[0][2] for d in dataset])
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.text(-4, 4, "Trained Agent Out of Dist")
    plt.show()

    visualize_value_fn(np.array([mdp.ACTION_UPDATES[i] for i in range(8)]), lambda x: agent.approximator.predict(x))


def plot_lspi_policies_linear(seed):
    eval_env = LinearSystem(discrete_actions=False, friction=0.1, full_init=True)
    feature_fun = LQRWeights(4, 2)
    new_policy = EpsGreedyPolicy.load(os.path.join("lqr_agents", "seed-%d.pkl") % seed)
    states, actions, rewards, next_states = rollout_policy(eval_env, new_policy, 42, n=20, visualize=True)


def plot_performances(ax, path, n_seeds, scale=1000, offset=0, color="C0"):
    performances = []
    for i in range(0, n_seeds):
        with open(os.path.join(path, "performances-%d.pkl" % i), "rb") as f:
            performances.append(pickle.load(f))

    mu = np.mean(performances, axis=0)
    se = np.std(performances, axis=0) / np.sqrt(len(performances))
    top = mu + 2 * se
    bottom = mu - 2 * se

    x = np.arange(0, mu.shape[0]) * scale + offset

    l, = ax.plot(x, mu, color=color)
    ax.fill_between(x, top, bottom, color=color, alpha=0.3)

    return l


def visualize_q_values(ticksize=6, fontsize=9, path=None):
    f = plt.figure(figsize=(2.3, 1.2))
    feature_fun = LQRWeights(4, 2)
    new_policy = EpsGreedyPolicy.load(os.path.join("logs", "lqr_agents", "seed-%d.pkl") % 0)
    ax = f.add_axes([0.09, 0.15, 0.41, 0.75])
    ax.set_title("LSPI", fontsize=0.8 * fontsize, pad=2.5)
    visualize_value_fn(ax, new_policy.actions, lambda x: np.einsum("...i,i->...", x, new_policy.w),
                       features=feature_fun)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.tick_params(axis='both', which='minor', labelsize=ticksize)

    mdp = LinearSystem(discrete_actions=True, friction=0.1, full_init=False)
    agent = BoostedDQN.load(os.path.join("logs", "dqn_agents_1000", "seed-%d.pkl" % 10))
    ax = f.add_axes([0.57, 0.15, 0.41, 0.75])
    ax.set_title("DQN", fontsize=0.8 * fontsize, pad=2.5)
    ax.set_yticklabels([])
    visualize_value_fn(ax, np.array([mdp.ACTION_UPDATES[i] for i in range(8)]), lambda x: agent.approximator.predict(x))
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.tick_params(axis='both', which='minor', labelsize=ticksize)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def visualize_performance(ticksize=6, fontsize=9, path=None):
    f = plt.figure(figsize=(3.5, 1.5))
    ax = f.add_axes([0.13, 0.2, 0.8, 0.67])

    twinax = ax.twinx()
    plot_performances(twinax, "logs/lqr_agents", 80, offset=0, scale=500, color="C0")
    plot_performances(twinax, "logs/dqn_agents_1000", 30, offset=0, scale=500, color="C2")
    twinax.set_xlim(ax.get_xlim())
    twinax.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    twinax.yaxis.offsetText.set_fontsize(ticksize)
    twinax.yaxis.get_offset_text().set_visible(False)
    twinax.text(41900, 300., r'$\times 10^{3}$', va='bottom', ha='center', size=ticksize)
    twinax.text(-2100, 300., r'$\times 10^{2}$', va='bottom', ha='center', size=ticksize)
    twinax.set_ylim([-5000, 0])
    twinax.tick_params(axis='both', which='major', labelsize=ticksize)
    twinax.tick_params(axis='both', which='minor', labelsize=ticksize)

    n_seeds = 100
    l1 = plot_performances(ax, "logs/boosted_agent_2000", n_seeds, offset=2000, color="C0")
    l2 = plot_performances(ax, "logs/curriculum_agent_25_1000", n_seeds, offset=10000, color="C2")
    l3 = plot_performances(ax, "logs/default_agent_200", n_seeds, offset=0, color="C3")
    l4 = plot_performances(ax, "logs/residual_agent_2000_25", n_seeds, offset=2000, color="C4")
    l5 = plot_performances(ax, "logs/ppr_agent_0.998_200", n_seeds, offset=2000, color="C5")
    l6 = plot_performances(ax, "logs/shaping_agent_4000", n_seeds, offset=2000, color="C6")
    l7 = plot_performances(ax, "logs/boosted_shaping_agent_4000", n_seeds, offset=2000, color="C7")
    ax.set_xlim(0, 40000)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(4, 4))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(2, 2))
    ax.yaxis.offsetText.set_fontsize(ticksize)
    ax.yaxis.get_offset_text().set_visible(False)
    ax.xaxis.offsetText.set_fontsize(ticksize)
    ax.set_xticks([0, 10000, 20000, 30000, 40000])
    ax.set_xlabel("Steps", fontsize=fontsize, labelpad=0.)
    ax.set_ylabel("Cum. Disc. Return", fontsize=fontsize, labelpad=0.)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.tick_params(axis='both', which='minor', labelsize=ticksize)

    ylim = ax.get_ylim()
    ax.axvline(2000, color='C0', alpha=.75, linestyle='--', linewidth=1.5)
    rect = patches.Rectangle((0, ylim[0]), 2000, ylim[1] - ylim[0], color='black', alpha=0.1)
    ax.add_patch(rect)
    ax.axvline(10000, color='C2', alpha=.75, linestyle='--', linewidth=1.5)
    rect = patches.Rectangle((0, ylim[0]), 10000, ylim[1] - ylim[0], color='black', alpha=0.1)
    ax.add_patch(rect)
    ax.set_ylim(ylim)

    ax.grid()
    ax.legend([l1, l2, l3, l4, l5, l6, l7], ["BC-DQN", "C-DQN", "DQN", "RPL", "PPR", "SC-DQN", "SBC-DQN"],
              fontsize=ticksize, ncol=7, loc=(-0.01, 1.02), columnspacing=0.5, handlelength=0.3, handletextpad=0.2)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    visualize_q_values(path="figures/linear_q_functions.pdf")
    visualize_performance(path="figures/linear_performances.pdf")
