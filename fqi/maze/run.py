import os
import time
import pickle
import pathlib
import argparse
import numpy as np
from joblib import Parallel, delayed
from fqi.fast_extra_tress import FastExtraTreesActionRegressor

from mushroom_rl.policy import EpsGreedy
from mushroom_rl.core import Core, Logger
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter

from fqi.fqi import BoostedFQI, FQI
from fqi.maze.puddle_world import PuddleWorld


def experiment(exp_id, curriculum, boosted):
    seed = 95 + exp_id
    np.random.seed(seed)
    print("Running with seed %d" % seed)

    alg = BoostedFQI if boosted else FQI

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment algorithm: ' + alg.__name__)

    n_iter = [10, 10, 10, 10, 20, 20, 50, 50, 50, 50]

    # MDP
    if curriculum:
        puddle_widths = [np.array([[.05, .05], [.05, .05]]),
                         np.array([[.05, .15], [.05, .15]]),
                         np.array([[.05, .25], [.05, .25]]),
                         np.array([[.05, .35], [.05, .35]]),
                         np.array([[.05, .45], [.05, .45]]),
                         np.array([[.05, .55], [.05, .55]]),
                         np.array([[.05, .65], [.05, .65]]),
                         np.array([[.05, .75], [.05, .75]]),
                         np.array([[.05, .85], [.05, .85]]),
                         np.array([[.05, .95], [.05, .95]])]
    else:
        puddle_widths = [np.array([[.05, .95], [.05, .95]])] * 10

    mdps = [PuddleWorld(horizon=200, gamma=.99, puddle_width=pw) for pw in puddle_widths]
    initial_opt_q = -47.557563488  # Optimal Q from initial state [0., 0.] of target task
    bins = np.linspace(mdps[0].info.observation_space.low[0], mdps[0].info.observation_space.high[0], 21)
    test_states = list()
    for i in bins:
        for j in bins:
            test_states.append([i, j])
    test_states = np.array(test_states)

    # Policy
    epsilon_values = [1., 1., 1., 1., .75, .75, .5, .5, .25, .25]
    epsilon = [Parameter(value=ev) for ev in epsilon_values]
    pi = EpsGreedy(epsilon=epsilon[0])
    test_epsilon = Parameter(value=0.)

    # Approximator
    approximator_params = dict(input_shape=mdps[0].info.observation_space.shape,
                               # output_shape=(mdps[0].info.action_space.n,),
                               n_actions=mdps[0].info.action_space.n,
                               n_models=len(mdps) if boosted else 1,
                               n_estimators=50,
                               min_samples_split=5,
                               min_samples_leaf=2,
                               random_state=seed)
    approximator_params['prediction'] = 'sum'

    algorithm_params = dict(n_iterations=1, quiet=True)

    # Agent
    agent = alg(mdps[0].info, pi, FastExtraTreesActionRegressor, approximator_params=approximator_params,
                **algorithm_params)

    js = list()
    qs = list()
    diff_qs = list()
    residuals = list()
    test_datasets = list()
    for i, mdp in enumerate(mdps):
        logger.info('TASK: %d\n-------' % i)

        # Algorithm
        core = Core(agent, mdp)

        # Dataset collection
        method_str = 'boosted_' if boosted else 'no_boosted_'
        method_str += 'curriculum' if curriculum else 'no_curriculum'
        dataset_path = os.path.join('data', method_str, 'dataset_%d_%d.pkl' % (i, exp_id))
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        try:
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
        except FileNotFoundError:
            agent.policy.set_epsilon(epsilon[i])
            logger.info('Generating dataset for task %d...' % i)
            t1 = time.time()
            dataset = core.evaluate(n_episodes=500, quiet=True)
            with open(dataset_path, 'wb') as f:
                pickle.dump(dataset, f)
            t2 = time.time()
            print("Dataset generation took %.3e" % (t2 - t1))

        if boosted:
            agent.set_curriculum_idx_and_reset(i)

        j_task = list()
        q_task = list()
        diff_q_task = list()
        residual_task = list()
        test_dataset_task = list()

        agent.policy.set_epsilon(test_epsilon)
        idx = np.arange(i + 1) if boosted else 0

        def test_callback():
            test_dataset = core.evaluate(n_episodes=1, quiet=True)
            j_task.append(np.mean(compute_J(test_dataset, mdp.info.gamma)))
            q_task.append(agent.approximator.predict(test_states, idx=idx))
            initial_qs = agent.approximator.predict(np.array([0., 0.]), idx=idx)
            diff_q_task.append(np.linalg.norm(initial_qs - initial_opt_q, ord=1) / len(initial_qs))
            test_dataset_task.append(test_dataset)
            if boosted and curriculum:
                residual_task.append(np.array(agent.approximator.predict(test_states, idx=idx[-1:])))

        t1 = time.time()
        agent.fit(dataset, callback=test_callback, n_iter=n_iter[i])
        t2 = time.time()

        js.append(j_task)
        qs.append(q_task)
        diff_qs.append(diff_q_task)
        test_datasets.append(test_dataset_task)
        if boosted and curriculum:
            residuals.append(residual_task)
        print("Training Time: %.3e" % (t2 - t1))

    return js, qs, diff_qs, residuals, test_datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-curriculum", action='store_true')
    parser.add_argument("--use-boosting", action='store_true')
    parser.add_argument("--n-exp", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=10)
    args = parser.parse_args()

    out = Parallel(n_jobs=args.n_jobs)(
        delayed(experiment)(i, args.use_curriculum, args.use_boosting) for i in range(args.n_exp))
    Js = [o[0] for o in out]
    Qs = [o[1] for o in out]
    diff_Qs = [o[2] for o in out]
    residuals = [o[3] for o in out]
    test_datasets = [o[4] for o in out]

    # Summary folder
    alg = 'boosted_' if args.use_boosting else 'no_boosted_'
    alg += 'curriculum' if args.use_curriculum else 'no_curriculum'
    folder_name = './logs/' + alg
    pathlib.Path(folder_name).mkdir(parents=True)

    np.save(folder_name + '/J.npy', Js)
    np.save(folder_name + '/Q.npy', Qs)
    np.save(folder_name + '/diff_Q.npy', diff_Qs)

    with open(folder_name + '/test_datasets.pkl', 'wb') as f:
        pickle.dump(test_datasets, f)

    if args.use_boosting and args.use_curriculum:
        np.save(folder_name + '/residuals.npy', residuals)
