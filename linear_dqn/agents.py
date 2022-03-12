import pickle
import numpy as np
from mushroom_rl.policy import TDPolicy
from mushroom_rl.utils.parameters import to_parameter


class LQRWeights:

    def __init__(self, state_dim, action_dim):
        self.full_dim = state_dim + action_dim
        self.indices = np.tril_indices(state_dim + action_dim)

    def __call__(self, xu):
        outers = np.einsum("...i,...j->...ij", xu, xu)
        squared_features = outers[..., self.indices[0], self.indices[1]]
        return squared_features

    def extract_lqr_weights(self, w):
        lower_mat = np.zeros((self.full_dim, self.full_dim))
        lower_mat[self.indices[0], self.indices[1]] = w
        return 0.5 * (lower_mat + lower_mat.T)


class LinearPolicy:

    def __init__(self, k):
        self.k = k

    def __call__(self, x):
        return np.einsum("ij,...j->...i", self.k, x)


class EpsGreedyPolicy:

    def __init__(self, eps, actions, feature_fun, w, discrete_action=False):
        self.eps = eps
        self.actions = actions
        self.feature_fun = feature_fun
        self.w = w
        self.discrete_action = discrete_action

    def compute_single_action(self, x):
        phi = self.feature_fun(
            np.concatenate((np.repeat(x[None, :], self.actions.shape[0], axis=0), self.actions), axis=1))
        qs = np.einsum("ij,j->i", phi, self.w)
        if self.discrete_action:
            return np.array([np.argmax(qs)])
        else:
            return self.actions[np.argmax(qs)]

    def predict(self, state, action=None):
        if action is None:
            # Only implemented for a batch of states - everything else would require serious numpy magic
            assert len(state.shape) == 2
            ext_states = np.repeat(state[None, :], self.actions.shape[0], axis=0)
            ext_actions = np.repeat(self.actions[:, None, :], state.shape[0], axis=1)
            phi = self.feature_fun(np.concatenate((ext_states, ext_actions), axis=-1))
            qs = np.einsum("...j,j->...", phi, self.w)
            return np.max(qs, axis=0)
        else:
            phi = self.feature_fun(np.concatenate((state, self.actions[np.squeeze(action)]), axis=-1))
            return np.einsum("...i,i->...", phi, self.w)

    def __call__(self, x):
        if np.random.uniform(0, 1) > self.eps:
            if len(x.shape) == 1:
                return self.compute_single_action(x)
            else:
                actions = []
                for x_single in x:
                    actions.append(self.compute_single_action(x_single))
                return np.array(actions)
        else:
            if len(x.shape) == 1:
                if self.discrete_action:
                    return np.array([np.random.randint(self.actions.shape[0])])
                else:
                    return self.actions[np.random.randint(self.actions.shape[0]), :]
            else:
                if self.discrete_action:
                    return np.random.randint(self.actions.shape[0], size=x.shape[0])
                else:
                    return np.squeeze(self.actions[np.random.randint(self.actions.shape[0], size=x.shape[0]), :])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.eps, self.actions, self.feature_fun, self.w), f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            eps, actions, feature_fun, w = pickle.load(f)
        return EpsGreedyPolicy(eps, actions, feature_fun, w)


class PPREpsGreedy(TDPolicy):
    """
    Epsilon greedy policy.

    """

    def __init__(self, epsilon, nu, prev_pi):
        """
        Constructor.

        Args:
            epsilon ([float, Parameter]): the exploration coefficient. It indicates
                the probability of performing a random actions in the current
                step.

        """
        super().__init__()

        self.prev_pi = prev_pi
        self._epsilon = to_parameter(epsilon)
        self._psi = 1.
        self._nu = nu
        self.progress_counts = True

        self._add_save_attr(_epsilon='mushroom')

    def __call__(self, *args):
        raise RuntimeError("Not implemented")

    def disable_count(self):
        self.progress_counts = False

    def enable_count(self):
        self.progress_counts = True

    def draw_action(self, state):
        if np.random.uniform() < self._psi:
            # Old policy prediction
            cur_action = self.prev_pi.compute_single_action(state)
        else:
            if self.progress_counts:
                eps = self._epsilon(state)
            else:
                eps = self._epsilon.get_value(state)

            # Epsilon-Greedy policy w.r.t the currently estimated Q-Function
            if np.random.uniform() < eps:
                cur_action = np.array([np.random.choice(self._approximator.n_actions)])
            else:
                q = self._approximator.predict(state, **self._predict_params)
                cur_action = np.argwhere(q == np.max(q)).ravel()

                if len(cur_action) > 1:
                    cur_action = np.array([np.random.choice(cur_action)])

        # Decay of old policy probability
        if self.progress_counts:
            self._psi *= self._nu

        return cur_action

    def set_epsilon(self, epsilon):
        """
        Setter.

        Args:
            epsilon ([float, Parameter]): the exploration coefficient. It indicates the
            probability of performing a random actions in the current step.

        """
        self._epsilon = to_parameter(epsilon)

    def update(self, *idx):
        """
        Update the value of the epsilon parameter at the provided index (e.g. in
        case of different values of epsilon for each visited state according to
        the number of visits).

        Args:
            *idx (list): index of the parameter to be updated.

        """
        self._epsilon.update(*idx)


class ResidualEpsGreedy(TDPolicy):
    """
    Epsilon greedy policy.

    """

    def __init__(self, epsilon, prev_pi, policy_offset=0):
        """
        Constructor.

        Args:
            epsilon ([float, Parameter]): the exploration coefficient. It indicates
                the probability of performing a random actions in the current
                step.

        """
        super().__init__()

        self.prev_pi = prev_pi
        self._epsilon = to_parameter(epsilon)
        self.offset_count = 0
        self.policy_increase = 1
        self.policy_offset = policy_offset

        self._add_save_attr(_epsilon='mushroom')

    def __call__(self, *args):
        raise RuntimeError("Not implemented")

    def disable_count(self):
        self.policy_increase = 0

    def enable_count(self):
        self.policy_increase = 1

    def draw_action(self, state):

        if np.random.uniform() < self._epsilon(state):
            cur_action = np.array([np.random.choice(self._approximator.n_actions)])
        else:
            # If an offset is specified - we only follow the old policy to gather more robust initial data
            if self.offset_count < self.policy_offset:
                self.offset_count += self.policy_increase
                cur_action = self.prev_pi.compute_single_action(state)
            else:
                q = self._approximator.predict(state, **self._predict_params)
                cur_action = np.argwhere(q == np.max(q)).ravel()

                if len(cur_action) > 1:
                    cur_action = np.array([np.random.choice(cur_action)])

        return cur_action

    def set_epsilon(self, epsilon):
        """
        Setter.

        Args:
            epsilon ([float, Parameter]): the exploration coefficient. It indicates the
            probability of performing a random actions in the current step.

        """
        self._epsilon = to_parameter(epsilon)

    def update(self, *idx):
        """
        Update the value of the epsilon parameter at the provided index (e.g. in
        case of different values of epsilon for each visited state according to
        the number of visits).

        Args:
            *idx (list): index of the parameter to be updated.

        """
        self._epsilon.update(*idx)
