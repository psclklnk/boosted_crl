from mushroom_rl.utils.spaces import *
from mushroom_rl.utils.viewer import Viewer
from mushroom_rl.core import Environment, MDPInfo


class PuddleWorld(Environment):
    """
    Puddle world as presented in:
    "Off-Policy Actor-Critic". Degris T. et al.. 2012.

    """

    def __init__(self, start=None, goal=None, goal_threshold=.1, noise_step=.0,
                 noise_reward=0., reward_goal=5., thrust=.05, puddle_center=None,
                 puddle_width=None, gamma=.99, horizon=5000):
        """
        Constructor.

        Args:
            start (np.array, None): starting position of the agent;
            goal (np.array, None): goal position;
            goal_threshold (float, .1): distance threshold of the agent from the
                goal to consider it reached;
            reward_goal (float, 0): reward obtained reaching goal state;
            thrust (float, .05): distance walked during each action;
            puddle_center (np.array, None): center of the puddle;
            puddle_width (np.array, None): width of the puddle;
            gamma (float, .99): discount factor.
            horizon (int, 5000): horizon of the problem;

        """
        # MDP parameters
        self._start = np.array([0., 0.]) if start is None else start
        self._goal = np.array([.95, .95]) if goal is None else goal
        self._goal_threshold = goal_threshold
        self._reward_goal = reward_goal
        self._thrust = thrust
        puddle_center = [[.2, .0], [.6, 1.]] if puddle_center is None else puddle_center
        self._puddle_center = np.array(puddle_center)
        puddle_width = [[.25, .55], [.25, .55]] if puddle_width is None else puddle_width
        self.puddle_width = np.array(puddle_width)

        self._actions = [np.zeros(2) for _ in range(4)]
        for i in range(4):
            self._actions[i][i // 2] = thrust * (i % 2 * 2 - 1)

        # MDP properties
        action_space = Discrete(4)
        observation_space = Box(0., 1., shape=(2,))
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        # Visualization
        self._pixels = None
        self._viewer = Viewer(1.0, 1.0)

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            self._state = self._start.copy()
        else:
            self._state = state

        return self._state

    def step(self, action):
        idx = action[0]
        next_state = self._state + self._actions[idx]
        self._state = next_state if not self._is_wall(next_state) else self._state
        self._state = np.clip(self._state, 0., 1.)

        absorbing = np.linalg.norm((self._state - self._goal),
                                   ord=1) < self._goal_threshold

        reward = -np.linalg.norm(self._goal - self._state, ord=1) if not absorbing else self._reward_goal

        return self._state, reward, absorbing, {}

    def render(self):
        if self._pixels is None:
            img_size = 100
            pixels = np.zeros((img_size, img_size, 3))
            for i in range(img_size):
                for j in range(img_size):
                    x = i / img_size
                    y = j / img_size
                    pixels[i, img_size - 1 - j] = self._is_wall(np.array([x, y]))

            pixels -= pixels.min()
            pixels *= 255. / pixels.max()
            self._pixels = np.floor(pixels)

        self._viewer.background_image(self._pixels)
        self._viewer.circle(self._state, 0.01,
                            color=(0, 255, 0))

        goal_area = [
            [-self._goal_threshold, 0],
            [0, self._goal_threshold],
            [self._goal_threshold, 0],
            [0, -self._goal_threshold]
        ]
        self._viewer.polygon(self._goal, 0, goal_area,
                             color=(255, 0, 0), width=1)

        self._viewer.display(0.1)

    def _is_wall(self, state):
        is_wall = 0
        for cen, wid in zip(self._puddle_center, self.puddle_width):
            if cen[0] - wid[0] < state[0] < cen[0] + wid[0] and cen[1] - wid[1] < state[1] < cen[1] + wid[1] \
                    and not np.allclose(cen[0] - wid[0], state[0]) and not np.allclose(cen[0] + wid[0], state[0]) \
                    and not np.allclose(cen[1] - wid[1], state[1]) and not np.allclose(cen[1] + wid[1], state[1]):
                is_wall = 1
                break

        return is_wall

    def stop(self):
        if self._viewer is not None:
            self._viewer.close()
