import argparse
import pathlib
import pickle

from joblib import Parallel, delayed
from fqi.fast_extra_tress import FastExtraTreesActionRegressor
from tqdm import trange
import numpy as np

from fqi.fqi import BoostedFQI
from fqi.car_on_hill.solver import solve_car_on_hill

from mushroom_rl.algorithms.value import FQI
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.car_on_hill import CarOnHill
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter


def experiment(exp_id, ms, boosted, iters_per_env):
    seed = 95 + exp_id
    np.random.seed(seed)
    print("Running with seed %d" % seed)

    alg = BoostedFQI if boosted else FQI

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment algorithm: ' + alg.__name__)

    # MDP
    mdps = [CarOnHill() for _ in range(len(ms))]
    for m, mdp in zip(ms, mdps):
        mdp._m = m
    n_tasks = len(mdps)

    names = ['%1.3f' % (mdp._m) for mdp in mdps]

    test_states_0 = np.linspace(mdps[0].info.observation_space.low[0],
                                mdps[0].info.observation_space.high[0], 10)
    test_states_1 = np.linspace(mdps[0].info.observation_space.low[1],
                                mdps[0].info.observation_space.high[1], 10)
    test_states = list()
    for s0 in test_states_0:
        for s1 in test_states_1:
            test_states += [s0, s1]
    test_states = np.array([test_states]).repeat(2, 0).reshape(-1, 2)
    test_actions = np.array(
        [np.zeros(len(test_states) // 2),
         np.ones(len(test_states) // 2)]).reshape(-1, 1).astype(np.int)

    # Test Q
    test_q = list()
    for i, mdp in enumerate(mdps):
        try:
            test_q.append(np.load('data/test_q_%s.npy' % names[i]).tolist())
        except FileNotFoundError:
            logger.info('Generating test Q-values for task %d...' % i)
            current_test_q = solve_car_on_hill(mdp, test_states, test_actions, mdp.info.gamma)
            pathlib.Path('data').mkdir(parents=True, exist_ok=True)
            np.save('data/test_q_%s.npy' % names[i], current_test_q)

            test_q.append(current_test_q)
    test_q = np.array(test_q)

    # Policy
    epsilon = Parameter(value=1.)
    test_epsilon = Parameter(value=0.)
    pi = EpsGreedy(epsilon=epsilon)

    # Approximator
    approximator_params = dict(input_shape=mdps[0].info.observation_space.shape,
                               n_actions=mdps[0].info.action_space.n,
                               n_models=n_tasks if boosted else 1,
                               n_estimators=50,
                               min_samples_split=5,
                               min_samples_leaf=2,
                               random_state=seed)
    approximator_params['prediction'] = 'sum'

    algorithm_params = dict(n_iterations=1)

    # Agent
    agent = alg(mdps[0].info, pi, FastExtraTreesActionRegressor, quiet=True, approximator_params=approximator_params,
                **algorithm_params)

    js = list()
    diff_qs = list()
    for i, mdp in enumerate(mdps):
        logger.info('TASK: %d\n-------' % i)
        if boosted:
            agent.set_curriculum_idx_and_reset(i)

        # Algorithm
        core = Core(agent, mdp)

        # Dataset collection
        try:
            with open('data/dataset_%1.3f.pkl' % mdp._m, 'rb') as f:
                dataset = pickle.load(f)
        except FileNotFoundError:
            agent.policy.set_epsilon(epsilon)
            logger.info('Generating dataset for task %d...' % i)
            dataset = core.evaluate(n_episodes=1000)
            with open('data/dataset_%1.3f.pkl' % mdp._m, 'wb') as f:
                pickle.dump(dataset, f)

        j_task = list()
        diff_q_task = list()
        agent.policy.set_epsilon(test_epsilon)
        idx = np.arange(i + 1) if boosted else 0
        # Loop
        for _ in trange(iters_per_env, dynamic_ncols=True, disable=False, leave=False):
            # Train
            agent.fit(dataset)

            # Test
            test_dataset = core.evaluate(initial_states=test_states, quiet=True)

            j_task.append(np.mean(compute_J(test_dataset, mdp.info.gamma)))
            qs = agent.approximator.predict(test_states, test_actions, idx=idx)
            diff_q_task.append(np.linalg.norm(qs - test_q[i], ord=1) / len(qs))

        js.append(j_task)
        diff_qs.append(diff_q_task)

    return js, diff_qs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-curriculum", action='store_true')
    parser.add_argument("--use-boosting", action='store_true')
    parser.add_argument("--n-exp", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=10)
    args = parser.parse_args()

    if args.use_curriculum:
        ms = [.8, 1., 1.2]
        iters_per_env = 20
    else:
        if args.use_boosting:
            ms = [1.2, 1.2, 1.2]
            iters_per_env = 20
        else:
            ms = [1.2]
            iters_per_env = 60

    out = Parallel(n_jobs=args.n_jobs)(
        delayed(experiment)(exp_id, ms, args.use_boosting, iters_per_env) for exp_id in range(args.n_exp))
    Js = [o[0] for o in out]
    Qs = [o[1] for o in out]

    # Summary folder
    alg = 'boosted' if args.use_boosting else 'no_boosted'
    cur = 'curriculum' if args.use_curriculum else 'no_curriculum'
    folder_name = './logs/' + alg + '_' + cur
    pathlib.Path(folder_name).mkdir(parents=True)
    np.save(folder_name + '/J.npy', Js)
    np.save(folder_name + '/Q.npy', Qs)

    print('J: ', np.mean(Js, 0))
    print('Q diff: ', np.mean(Qs, 0))
