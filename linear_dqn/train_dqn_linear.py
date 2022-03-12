from joblib import Parallel, delayed

import os
import torch
import pickle
import argparse
import numpy as np
import torch.optim as optim
from linear_dqn.dqn import DQN
import torch.nn.functional as F
from mushroom_rl.core import Core
from linear_dqn.network import Network
from linear_dqn.envs import LinearSystem
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.parameters import LinearParameter, Parameter

torch.set_num_threads(1)


def train_dqn(seed, path, performance_path, target_update_frequency=1000):
    np.random.seed(seed)
    friction = 0.1
    mdp = LinearSystem(discrete_actions=True, friction=friction, full_init=False)

    if not os.path.exists(path):
        optimizer = {'class': optim.Adam, 'params': dict(lr=1e-4, eps=1e-8)}

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

        initial_replay_size = 500
        algorithm_params = dict(
            batch_size=32,
            target_update_frequency=target_update_frequency,
            initial_replay_size=initial_replay_size,
            max_replay_size=10000
        )

        pi = EpsGreedy(Parameter(value=1.0))
        agent = DQN(mdp.info, pi, TorchApproximator, approximator_params=approximator_params, **algorithm_params)
        core = Core(agent, mdp)

        performances = [np.mean(compute_J(core.evaluate(n_episodes=50), gamma=mdp.info.gamma))]
        core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)
        agent_parameter = LinearParameter(value=1.0, threshold_value=0.1, n=1000)
        pi.set_epsilon(agent_parameter)
        n = 20
        for i in range(0, n):
            core.learn(n_steps=500, n_steps_per_fit=1)
            pi.set_epsilon(Parameter(value=0.05))
            performances.append(np.mean(compute_J(core.evaluate(n_episodes=50), gamma=mdp.info.gamma)))
            pi.set_epsilon(agent_parameter)

        agent.save(path)

        with open(performance_path, "wb") as f:
            pickle.dump(np.array(performances), f)


def experiment(seed, target_update_frequency):
    log_dir = "logs/dqn_agents_%d" % target_update_frequency
    os.makedirs(log_dir, exist_ok=True)
    train_dqn(seed, os.path.join(log_dir, "seed-%d.pkl" % seed), os.path.join(log_dir, "performances-%d.pkl" % seed),
              target_update_frequency=target_update_frequency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--n-exp", type=int, default=100)
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    Parallel(n_jobs=args.n_jobs)(delayed(experiment)(k, 1000) for k in range(args.n_exp))
