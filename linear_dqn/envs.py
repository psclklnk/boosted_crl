import pygame
import numpy as np
from mushroom_rl.utils import spaces
from mushroom_rl.utils.viewer import Viewer
from mushroom_rl.core import Environment, MDPInfo


class ShapingReward:

    def __init__(self, gamma, value_function):
        self.gamma = gamma
        self.value_function = value_function

    def __call__(self, state, next_state):
        values = self.value_function.predict(np.stack([state, next_state], axis=0))
        return self.gamma * values[1] - values[0]


class LinearSystem(Environment):
    ACTION_UPDATES = {
        # 0 = UP
        0: np.array([1., 0.]),
        # 1 = RIGHT
        1: np.array([0., 1.]),
        # 2 = DOWN
        2: np.array([-1., 0.]),
        # 3 = LEFT
        3: np.array([0., -1.]),
        # 4 = UP/RIGHT
        4: np.array([1., 1.]) / np.sqrt(2),
        # 5 = UP/LEFT
        5: np.array([1., -1.]) / np.sqrt(2),
        # 6 = DOWN/RIGHT
        6: np.array([-1., 1.]) / np.sqrt(2),
        # 7 = DOWN/LEFT
        7: np.array([-1., -1.]) / np.sqrt(2)
    }

    def __init__(self, horizon=100, gamma=.99, friction=1., dt=0.05, discrete_actions=True, full_init=False):
        self.state = None
        self.goal = np.zeros(4)
        self.weights = np.array([3., 1., 3., 1.])
        self.action_weights = np.array([1e-1, 1e-1])
        self.discrete_actions = discrete_actions

        observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
                                       high=np.array([np.inf, np.inf, np.inf, np.inf]))
        action_space = spaces.Discrete(8)
        # self.viewer = ImageViewer((450, 450), 0.1)
        self.sys_mat = np.array([[0., 1., 0., 0.],
                                 [0., -friction, 0., 0],
                                 [0., 0., 0., 1.],
                                 [0., 0., 0., -friction]])
        self.c_mat = np.array([[0., 0.],  # was 10
                               [2., 0.],
                               [0., 0.],
                               [0., 2.]])
        self.dt = dt
        self.full_init = full_init

        super().__init__(MDPInfo(observation_space, action_space, gamma, horizon))

    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self, a):
        if self.discrete_actions:
            if isinstance(a, np.ndarray):
                a = a[0]
            a = self.ACTION_UPDATES[a]

        # Simulate the system
        new_state = self.state + self.dt * (np.dot(self.sys_mat, self.state) + np.dot(self.c_mat, a) +
                                            np.random.normal(np.zeros(4), np.array([1e-5, 1e-5, 1e-5, 1e-5])))
        self.state = new_state

        return np.copy(self.state), \
               -np.sum(self.weights * np.square(self.goal - self.state)) - np.sum(
                   self.action_weights * np.square(a)), False, {}

    def reset(self, state=None):
        if self.full_init:
            self.state = np.random.uniform(-4, 4, size=(4,))
            self.state[1::2] = np.random.uniform(-3e-1, 3e-1, size=(2,))
        else:
            self.state = np.array([-4., 0., 0., 0.]) + np.random.uniform(-3e-1, 3e-1, size=(4,))

        return np.copy(self.state)

    def render(self, mode='human'):
        None


class Wall:

    def __init__(self, vector, blocks_forward=True, blocks_backward=True):
        self.normal_vector = vector / np.linalg.norm(vector)
        self.offset = np.linalg.norm(vector)
        self.blocks_foward = blocks_forward
        self.blocks_backward = blocks_backward

    def process_collision(self, start_state, end_state, with_collision_info=False):
        start_projection = np.dot(start_state[0::2], self.normal_vector)
        end_projection = np.dot(end_state[0::2], self.normal_vector)

        if (self.blocks_foward and start_projection < self.offset and end_projection >= self.offset) or \
                (self.blocks_backward and start_projection > self.offset and end_projection <= self.offset):
            collision_time = (self.offset - start_projection) / (end_projection - start_projection)

            if with_collision_info:
                # Flip on part of the vector and integrate for the remaining time
                intermediate_state = start_state + collision_time * (end_state - start_state)

                vel_projection = np.dot(intermediate_state[1::2], self.normal_vector)
                intermediate_state[1::2] -= 2 * vel_projection * self.normal_vector

                return collision_time, intermediate_state
            else:
                return collision_time
        else:
            if with_collision_info:
                return np.inf, None
            else:
                return np.inf


class NonLinearSystem(Environment):
    ACTION_UPDATES = {
        # 0 = RIGHT
        0: np.array([1., 0.]),
        # 1 = UP
        1: np.array([0., 1.]),
        # 2 = LEFT
        2: np.array([-1., 0.]),
        # 3 = DOWN
        3: np.array([0., -1.]),
        # 4 = RIGHT/UP
        4: np.array([1., 1.]) / np.sqrt(2),
        # 5 = RIGHT/DOWN
        5: np.array([1., -1.]) / np.sqrt(2),
        # 6 = RIGHT/UP
        6: np.array([-1., 1.]) / np.sqrt(2),
        # 7 = LEFT/DOWN
        7: np.array([-1., -1.]) / np.sqrt(2)
    }

    def __init__(self, horizon=150, gamma=.99, friction=1., dt=0.05, discrete_actions=True):
        self.state = None
        self.goal = np.zeros(4)
        self.discrete_actions = discrete_actions

        observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
                                       high=np.array([np.inf, np.inf, np.inf, np.inf]))
        action_space = spaces.Discrete(8)
        # self.viewer = ImageViewer((450, 450), 0.1)
        self.sys_mat = np.array([[0., 1., 0., 0.],
                                 [0., -friction, 0., 0],
                                 [0., 0., 0., 1.],
                                 [0., 0., 0., -friction]])
        self.c_mat = np.array([[0., 0.],  # was 10
                               [2., 0.],
                               [0., 0.],
                               [0., 2.]])
        self.dt = dt
        self._viewer = Viewer(6, 2, background=(255, 255, 255), width=600, height=200)
        self.path_width = 0.5

        self.walls = [Wall(np.array([-5., 0.])),
                      Wall(np.array([1., 0.])),
                      Wall(np.array([0., 3.])),
                      Wall(np.array([0., -3.])),
                      Wall(np.array([0., self.path_width]), blocks_forward=False),
                      Wall(np.array([0., -self.path_width]), blocks_forward=False)]
        self.prev_q = None
        self.shaping_reward = None

        super().__init__(MDPInfo(observation_space, action_space, gamma, horizon))

    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self, a):
        if self.discrete_actions:
            if self.prev_q is not None:
                a = combine_actions(a, self.prev_q(self.state))

            if isinstance(a, np.ndarray):
                a = a[0]
            a = self.ACTION_UPDATES[a]
        else:
            a += self.prev_q(self.state)

        # In lower half:
        ext_f = np.array([-0.5, 0., 0., 0.])
        f = np.dot(self.sys_mat, self.state) + np.dot(self.c_mat, a) + ext_f + \
            np.random.normal(np.zeros(4), np.array([1e-5, 1e-5, 1e-5, 1e-5]))

        # Simulate the system
        new_state = self.state + self.dt * f
        dt = self.dt
        while dt > 1e-5:
            collision_times = np.array([w.process_collision(self.state, new_state) for w in self.walls])
            collision_idx = np.argmin(collision_times)

            if collision_times[collision_idx] == np.inf:
                break
            else:
                intermediate_state = self.walls[collision_idx].process_collision(self.state, new_state,
                                                                                 with_collision_info=True)[1]
                dt = dt * (1 - collision_times[collision_idx])
                old_state = intermediate_state
                new_state = old_state + dt * np.dot(self.sys_mat, old_state) + \
                            np.random.normal(np.zeros(4), np.array([1e-5, 1e-5, 1e-5, 1e-5]))
        if self.shaping_reward is not None:
            sr = self.shaping_reward(self.state, new_state)
        else:
            sr = 0.

        self.state = new_state

        if not (-self.path_width <= self.state[2] <= self.path_width):
            self.reset()

        success = np.linalg.norm(self.state[0::2]) < 0.2
        reward = sr + (0 if success else -1.)

        return np.copy(self.state), reward, success, {}

    def reset(self, state=None):
        self.state = np.array([-4., 0., 0., 0.])
        noise_vec = np.array([1e-3, 1e-1, 4e-1, 1e-1])
        self.state += np.random.uniform(-noise_vec, noise_vec)

        return np.copy(self.state)

    def draw_custom_arrow(self, center, direction, force, max_force, max_length, color=(255, 255, 255),
                          width=1, head_scale=1.):
        length = force / max_force * max_length

        if length != 0:
            c = self._viewer._transform(center)
            direction = direction / np.linalg.norm(direction)
            end = center + length * direction
            e = self._viewer._transform(end)
            delta = e - c

            pygame.draw.line(self._viewer.screen, color, c, e, width)
            self._viewer.arrow_head(end, head_scale, -np.arctan2(delta[1], delta[0]), color)

    def render(self, mode='human'):
        # Constant offset
        offset = np.array([5., 1.])

        # Car
        pit1 = np.array([
            [-10, -self.path_width],
            [10, -self.path_width],
            [10, -10],
            [-10, -10]])

        pit2 = np.array([
            [-10, self.path_width],
            [10, self.path_width],
            [10, 10],
            [-10, 10]])

        self._viewer.polygon(np.zeros(2), 0., pit1 + offset[None, :], color=(180, 180, 180))
        self._viewer.polygon(np.zeros(2), 0., pit2 + offset[None, :], color=(180, 180, 180))

        for x in np.linspace(-4.7, 2, 12):
            for y in np.linspace(-0.75 * self.path_width, 0.75 * self.path_width, 3):
                self.draw_custom_arrow(np.array([x, y]) + offset, np.array([-1., 0.]),
                                       0.5, 1., 0.5, color=(100, 100, 100), width=8, head_scale=0.2)

        self._viewer.circle(self.state[0::2] + offset, 0.15, color=(0, 0, 0))
        self._viewer.circle(np.zeros(2) + offset, 0.15, color=(255, 0, 0))

        self._viewer.display(self.dt)


def combine_actions(cur_action, prev_action):
    # Combine the actions and project the resulting one to the closes allowed one
    cont_actions = np.array([
        [1., 0.],
        [0., 1.],
        [-1., 0.],
        [0., -1.],
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        [1 / np.sqrt(2), -1 / np.sqrt(2)],
        [-1 / np.sqrt(2), 1 / np.sqrt(2)],
        [-1 / np.sqrt(2), -1 / np.sqrt(2)]
    ])

    comb_action = cont_actions[cur_action] + cont_actions[prev_action]
    if np.all(np.isclose(comb_action, np.zeros(2))):
        # In this case we take the action selected by the current policy (as the two actions cancel out each
        # other)
        return cur_action
    else:
        action_dists = np.linalg.norm(comb_action - cont_actions, axis=-1)
        candidates = np.where(np.isclose(action_dists, np.min(action_dists)))[0]

        if candidates.shape[0] > 1:
            # The combination in the discrete case can yield two candidates. In this case we choose the one closer to
            # the action issued by the new agent
            chosen_action = candidates[
                np.argmin(np.linalg.norm(cont_actions[cur_action] - cont_actions[candidates], axis=-1))]
            return np.array([chosen_action])
        else:
            return candidates
