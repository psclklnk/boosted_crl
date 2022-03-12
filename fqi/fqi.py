import numpy as np
from tqdm import trange

from mushroom_rl.algorithms.value.batch_td import BatchTD
from mushroom_rl.utils.dataset import parse_dataset


class BoostedFQI(BatchTD):
    def __init__(self, mdp_info, policy, approximator, n_iterations,
                 approximator_params=None, fit_params=None, quiet=False):
        self._n_iterations = n_iterations
        self._quiet = quiet
        self._curriculum_idx = None

        self._add_save_attr(
            _n_iterations='primitive',
            _quiet='primitive',
            _target='pickle',
            _curriculum_idx='primitive'
        )

        super().__init__(mdp_info, policy, approximator, approximator_params, fit_params)

    def fit(self, x, callback=None, n_iter=None):
        if n_iter is None:
            n_iter = self._n_iterations

        state, action, reward, next_state, absorbing, _ = parse_dataset(x)

        if self._curriculum_idx > 0:
            prev_q = self.approximator.predict(state, action.astype(np.int64), idx=np.arange(self._curriculum_idx))
            prev_q_next_state = self.approximator.predict(next_state, idx=np.arange(self._curriculum_idx))
        else:
            prev_q = np.zeros(state.shape[0])
            prev_q_next_state = np.zeros((next_state.shape[0], self.approximator.n_actions))

        for _ in trange(n_iter, dynamic_ncols=True, disable=self._quiet, leave=False):
            q_next_state = prev_q_next_state + self.approximator.predict(next_state, idx=self._curriculum_idx)

            if np.any(absorbing):
                q_next_state *= 1 - absorbing.reshape(-1, 1)

            max_q_next_state = np.max(q_next_state, axis=1)
            target = reward + self.mdp_info.gamma * max_q_next_state - prev_q
            self.approximator.fit(state, action, target, idx=self._curriculum_idx)

            if callback is not None:
                callback()

    def set_curriculum_idx_and_reset(self, curriculum_idx):
        self._curriculum_idx = curriculum_idx
        self.policy._predict_params['idx'] = np.arange(curriculum_idx + 1)


class FQI(BatchTD):
    """
    This is the mushroom implementation with the simple difference that the number of iterations can be set by a method
    and that the fit method takes a callback that it calls after every iteration

    """

    def __init__(self, mdp_info, policy, approximator, n_iterations,
                 approximator_params=None, fit_params=None, quiet=False):
        """
        Constructor.

        Args:
            n_iterations ([int, Parameter]): number of iterations to perform for training;
            quiet (bool, False): whether to show the progress bar or not.

        """
        self._n_iterations = n_iterations
        self._quiet = quiet
        self._target = None

        self._add_save_attr(
            _n_iterations='mushroom',
            _quiet='primitive',
            _target='pickle'
        )

        super().__init__(mdp_info, policy, approximator, approximator_params, fit_params)

    def fit(self, x, callback=None, n_iter=None):
        if n_iter is None:
            n_iter = self._n_iterations

        state, action, reward, next_state, absorbing, _ = parse_dataset(x)
        for _ in trange(n_iter, dynamic_ncols=True, disable=self._quiet, leave=False):
            if self._target is None:
                self._target = reward
            else:
                q = self.approximator.predict(next_state)
                if np.any(absorbing):
                    q *= 1 - absorbing.reshape(-1, 1)

                max_q = np.max(q, axis=1)
                self._target = reward + self.mdp_info.gamma * max_q

            self.approximator.fit(state, action, self._target, **self._fit_params)

            if callback is not None:
                callback()
