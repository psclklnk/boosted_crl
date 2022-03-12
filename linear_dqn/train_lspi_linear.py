from joblib import Parallel, delayed

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from linear_dqn.agents import LQRWeights, LinearPolicy, EpsGreedyPolicy
from linear_dqn.envs import LinearSystem


def lstdq(features, states, actions, rewards, next_states, next_actions, gamma=0.99):
    phi_state_action = features(np.concatenate((states, actions), axis=-1))
    phi_next_state_action = features(np.concatenate((next_states, next_actions), axis=-1))

    tmp = phi_state_action - gamma * phi_next_state_action

    A = np.einsum("nj,nk->jk", phi_state_action, tmp)
    b = np.einsum("nj,n->j", phi_state_action, rewards)

    if np.linalg.matrix_rank(A) == A.shape[1]:
        w = np.linalg.solve(A, b).ravel()
    else:
        w = np.linalg.pinv(A).dot(b).ravel()

    return w


def rollout_policy(env, policy, i, n=2, visualize=False):
    # Generate some initial data
    states = []
    actions = []
    rewards = []
    next_states = []
    for _ in range(0, n):
        state_buffer = []
        action_buffer = []
        reward_buffer = []

        state = env.reset()
        for _ in range(0, 100):
            state_buffer.append(state)
            action = policy(state) + np.random.normal(0, 0.1, size=(2,))
            action_buffer.append(action)
            state, reward, done, info = env.step(action)
            reward_buffer.append(reward)

        state_buffer = np.array(state_buffer)
        action_buffer = np.array(action_buffer)
        reward_buffer = np.array(reward_buffer)

        states.append(state_buffer[:-1])
        actions.append(action_buffer[:-1])
        rewards.append(reward_buffer[:-1])
        next_states.append(state_buffer[1:])

    states = np.concatenate(states, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    next_states = np.concatenate(next_states, axis=0)

    if visualize:
        plt.scatter(states[:, 0], states[:, 2])
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.text(-3.5, 3.5, "Iteration %d" % i)
        plt.show()

    return states, actions, rewards, next_states


def evaluate_policy(env, policy, n=50):
    rewards = []
    for _ in range(n):
        cum_rew = 0.
        disc = 1.
        s = env.reset()
        for i in range(env.info.horizon):
            s, r, __, __ = env.step(policy(s))
            cum_rew += disc * r
            disc *= env.info.gamma
        rewards.append(cum_rew)
    return np.mean(rewards)


def train_agent(seed, policy_path, performance_path):
    np.random.seed(seed)
    feature_fun = LQRWeights(4, 2)
    friction = 0.1  # was 0.1
    env = LinearSystem(discrete_actions=False, friction=friction)
    eval_env = LinearSystem(discrete_actions=False, friction=friction)

    k_init = np.array([[0., 0, 0., 0.],
                       [0., 0., 0., 0.]])
    policy = LinearPolicy(k_init)

    if not os.path.exists(policy_path):
        performances = [evaluate_policy(eval_env, policy)]
        for i in range(0, 4):
            states, actions, rewards, next_states = rollout_policy(env, policy, i, visualize=False, n=5)
            w = lstdq(feature_fun, states, actions, rewards, next_states, policy(next_states))

            if isinstance(policy, EpsGreedyPolicy):
                policy = EpsGreedyPolicy(1. - (i + 1) * 0.25, np.array([env.ACTION_UPDATES[i] for i in range(8)]),
                                         feature_fun, w)
            else:
                q = feature_fun.extract_lqr_weights(w)

                # Sanity check that the matrix encodes the same behaviour
                if policy.k[0, 0] <= 0 and policy.k[1, 2] <= 0:
                    old_pred = np.einsum("...j,j->...", feature_fun(np.concatenate((states, actions), axis=-1)), w)
                    new_pred = np.einsum("...i,ij,...j->...", np.concatenate((states, actions), axis=-1), q,
                                         np.concatenate((states, actions), axis=-1))
                    assert np.all(np.abs(old_pred - new_pred) < 1e-10)

                    policy.k = -np.linalg.solve(q[4:, 4:], q[4:, :4])
                    print("Iteration %d" % (i + 1))
                    eigvals, eigvecs = np.linalg.eig(q)
                    print(eigvals)
                else:
                    raise RuntimeError("Policy became unstable")

            performances.append(evaluate_policy(eval_env, policy))

        new_policy = EpsGreedyPolicy(0.05, np.array([env.ACTION_UPDATES[i] for i in range(8)]), feature_fun, w)
        new_policy.save(policy_path)
        with open(performance_path, "wb") as f:
            pickle.dump(np.array(performances), f)


def experiment(seed):
    os.makedirs("logs/lqr_agents", exist_ok=True)
    train_agent(seed, os.path.join("logs", "lqr_agents", "seed-%d.pkl" % seed),
                os.path.join("logs", "lqr_agents", "performances-%d.pkl" % seed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--n-exp", type=int, default=100)
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    Parallel(n_jobs=args.n_jobs)(delayed(experiment)(k) for k in range(args.n_exp))
