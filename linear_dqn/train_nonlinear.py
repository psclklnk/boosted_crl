from joblib import Parallel, delayed

import os
import pickle
import argparse
import numpy as np
from pynput import keyboard
from queue import Queue, Empty

from linear_dqn.network import Network
from linear_dqn.dqn import SingleBoostedDQN, DQN
from linear_dqn.envs import NonLinearSystem, ShapingReward
from linear_dqn.agents import EpsGreedyPolicy, PPREpsGreedy, ResidualEpsGreedy

from mushroom_rl.core import Core
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.parameters import Parameter, LinearParameter

import torch
import torch.optim as optim
import torch.nn.functional as F

torch.set_num_threads(1)


def inspect_env():
    mdp = NonLinearSystem()

    queue = Queue(maxsize=1)

    def on_release(key):
        print(key)
        if key == keyboard.Key.down:
            queue.put(3, block=True)
        elif key == keyboard.Key.right:
            queue.put(0, block=True)
        elif key == keyboard.Key.up:
            queue.put(1, block=True)
        elif key == keyboard.Key.left:
            queue.put(2, block=True)
        elif key == keyboard.Key.esc:
            queue.put(None, block=True)

    listener = keyboard.Listener(
        on_release=on_release)
    listener.start()

    for i in range(0, 1):
        done = False
        mdp.reset()
        mdp.render()
        last_action = np.array([0])
        while not done:
            try:
                last_action = queue.get(block=False)
            except Empty:
                pass
            state, reward, done, info = mdp.step(last_action)
            print((state, reward))
            mdp.render()

    listener.stop()


def create_agent(mdp, pi, prev_q=None, initial_replay_size=500, target_update_frequency=1000, learning_rate=1e-4):
    optimizer = {'class': optim.Adam, 'params': dict(lr=learning_rate, eps=1e-8)}

    approximator_params = dict(
        network=Network,
        input_shape=mdp.info.observation_space.shape,
        output_shape=(mdp.info.action_space.n,),
        n_actions=mdp.info.action_space.n,
        n_features=128,
        two_layers=True,
        loss=F.smooth_l1_loss,
        optimizer=optimizer,
        use_cuda=False
    )

    initial_replay_size = initial_replay_size
    algorithm_params = dict(
        batch_size=32,
        target_update_frequency=target_update_frequency,
        initial_replay_size=initial_replay_size,
        max_replay_size=10000
    )

    if prev_q is None:
        agent = DQN(mdp.info, pi, TorchApproximator, approximator_params=approximator_params,
                    **algorithm_params)
    else:
        agent = SingleBoostedDQN(prev_q, mdp.info, pi, TorchApproximator, approximator_params=approximator_params,
                                 **algorithm_params)

    return agent


def create_default_agent(mdp, seed, target_update_frequency):
    pi = EpsGreedy(Parameter(1.))
    agent = create_agent(mdp, pi, prev_q=None, initial_replay_size=500, target_update_frequency=target_update_frequency)
    core = Core(agent, mdp)
    core.learn(n_steps=500, n_steps_per_fit=500)
    pi.set_epsilon(LinearParameter(value=1.0, threshold_value=0.1, n=1000))
    return core


def create_boosted_agent(mdp, seed, target_update_frequency, with_shaping=False):
    prev_q = EpsGreedyPolicy.load(os.path.join("logs/lqr_agents", "seed-%d.pkl" % seed))
    prev_q.discrete_action = True
    if with_shaping:
        mdp.shaping_reward = ShapingReward(mdp.info.gamma, prev_q)

    pi = EpsGreedy(LinearParameter(value=0.1, threshold_value=0.1, n=1))
    agent = create_agent(mdp, pi, prev_q=prev_q, initial_replay_size=500,
                         target_update_frequency=target_update_frequency)
    return Core(agent, mdp)


def create_curriculum_agent(mdp, seed, target_update_frequency, overwrite_target_update_frequency=None):
    agent = DQN.load(os.path.join("logs/dqn_agents_%d" % target_update_frequency, "seed-%d.pkl" % seed))
    agent._replay_memory.reset()
    if overwrite_target_update_frequency is not None:
        agent._target_update_frequency = overwrite_target_update_frequency
    agent._curriculum_idx = 0
    return Core(agent, mdp)


def create_shaping_agent(mdp, seed, target_update_frequency):
    prev_q = EpsGreedyPolicy.load(os.path.join("logs/lqr_agents", "seed-%d.pkl" % seed))
    prev_q.discrete_action = True
    mdp.shaping_reward = ShapingReward(mdp.info.gamma, prev_q)

    return create_default_agent(mdp, seed, target_update_frequency=target_update_frequency)


def create_residual_agent(mdp, seed, target_update_frequency, policy_offset=0):
    prev_q = EpsGreedyPolicy.load(os.path.join("logs/lqr_agents", "seed-%d.pkl" % seed))
    prev_q.discrete_action = True
    prev_q.eps = 0.
    mdp.prev_q = prev_q
    pi = ResidualEpsGreedy(LinearParameter(value=0.1, threshold_value=0.1, n=1), prev_q, policy_offset=policy_offset)
    agent = create_agent(mdp, pi, prev_q=None, initial_replay_size=500, target_update_frequency=target_update_frequency)
    return Core(agent, mdp)


def create_ppr_agent(mdp, seed, target_update_frequency, decay_factor):
    prev_q = EpsGreedyPolicy.load(os.path.join("logs/lqr_agents", "seed-%d.pkl" % seed))
    prev_q.discrete_action = True
    prev_q.eps = 0.
    pi = PPREpsGreedy(LinearParameter(value=0.1, threshold_value=0.1, n=1), decay_factor, prev_q)
    agent = create_agent(mdp, pi, prev_q=None, initial_replay_size=500, target_update_frequency=target_update_frequency)
    return Core(agent, mdp)


def learn(core, n_steps, steps_per_eval, n_eval_steps, save_path):
    assert n_steps % steps_per_eval == 0
    sr = core.mdp.shaping_reward

    if isinstance(core.agent.policy, ResidualEpsGreedy) or isinstance(core.agent.policy, PPREpsGreedy):
        core.agent.policy.disable_count()
    core.mdp.shaping_reward = None
    performances = [np.mean(compute_J(core.evaluate(n_steps=n_eval_steps)))]
    if isinstance(core.agent.policy, ResidualEpsGreedy) or isinstance(core.agent.policy, PPREpsGreedy):
        core.agent.policy.enable_count()
    core.mdp.shaping_reward = sr
    for i in range(n_steps // steps_per_eval):
        core.learn(n_steps=steps_per_eval, n_steps_per_fit=1)
        if isinstance(core.agent.policy, ResidualEpsGreedy) or isinstance(core.agent.policy, PPREpsGreedy):
            core.agent.policy.disable_count()
        core.mdp.shaping_reward = None
        performances.append(np.mean(compute_J(core.evaluate(n_steps=n_eval_steps))))
        if isinstance(core.agent.policy, ResidualEpsGreedy) or isinstance(core.agent.policy, PPREpsGreedy):
            core.agent.policy.enable_count()
        core.mdp.shaping_reward = sr
    core.agent.save(save_path)
    return performances


def run_agent(seed, save_dir, target_update_frequency, n_steps, agent_fn, **agent_params):
    save_dir = save_dir + ("_%d" % target_update_frequency)
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(os.path.join(save_dir, "agent-%d.pkl" % seed)) or not os.path.exists(
            os.path.join(save_dir, "performances-%d.pkl" % seed)):
        mdp = NonLinearSystem(discrete_actions=True)
        np.random.seed(seed)
        core = agent_fn(mdp, seed, target_update_frequency, **agent_params)
        performances = learn(core, n_steps, 1000, 5000, os.path.join(save_dir, "agent-%d.pkl" % seed))
        with open(os.path.join(save_dir, "performances-%d.pkl" % seed), "wb") as f:
            pickle.dump(performances, f)


def run_default_agent(seed, save_dir, target_update_frequency):
    run_agent(seed, save_dir, target_update_frequency, 40000, create_default_agent)


def run_residual_agent(seed, save_dir, target_update_frequency, policy_offset=0):
    run_agent(seed, save_dir, target_update_frequency, 38000, create_residual_agent, policy_offset=policy_offset)


def run_ppr_agent(seed, save_dir, target_update_frequency, decay_factor):
    run_agent(seed, save_dir, target_update_frequency, 38000, create_ppr_agent, decay_factor=decay_factor)


def run_boosted_agent(seed, save_dir, target_update_frequency, with_shaping=False):
    run_agent(seed, save_dir, target_update_frequency, 38000, create_boosted_agent, with_shaping=with_shaping)


def run_curriculum_agent(seed, save_dir, target_update_frequency, overwrite_target_update_frequency=None):
    run_agent(seed, save_dir, target_update_frequency, 30000, create_curriculum_agent,
              overwrite_target_update_frequency=overwrite_target_update_frequency)


def run_shaping_agent(seed, save_dir, target_update_frequency):
    run_agent(seed, save_dir, target_update_frequency, 38000, create_shaping_agent)


def experiment(seed):
    run_shaping_agent(seed, "logs/shaping_agent", 4000)
    run_ppr_agent(seed, "logs/ppr_agent_0.998", 200, 0.998)
    run_residual_agent(seed, "logs/residual_agent_2000", 25, policy_offset=2000)
    run_boosted_agent(seed, "logs/boosted_agent", 2000)
    run_boosted_agent(seed, "logs/boosted_shaping_agent", 4000, with_shaping=True)
    run_default_agent(seed, "logs/default_agent", 200)
    run_curriculum_agent(seed, "logs/curriculum_agent_25", 1000, overwrite_target_update_frequency=25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--n-exp", type=int, default=100)
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    Parallel(n_jobs=args.n_jobs)(delayed(experiment)(k) for k in range(args.n_exp))
