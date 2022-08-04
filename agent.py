import gym
import numpy as np


class MountainCarContinuousAgent(object):
    """
    This class solves the Continuous Mountain Car problem using
    fuzzy logic.
    """

    def __init__(self, gamma=1.0, alpha=0.1):
        self.env = gym.make("MountainCarContinuous-v0")
        self.alpha = alpha
        self.gamma = gamma
        # we have 64 (8 * 8) fuzzy states and two discrete actions
        self.q = np.zeros((64, 2))
        self.e = np.zeros((64, 2))

    def select_actions(self):
        """
        selects an action for each fuzzy state
        :return: ndarray. index of chosen actions.
        """
        actions = []
        for i in range(64):
            values = self.q[i]
            softmax = np.exp(values - np.max(values))
            softmax /= np.sum(softmax)
            action = np.random.choice(2, p=softmax)
            actions.append(action)
        return np.array(actions)

    @staticmethod
    def realize_actions(actions):
        """
        converts indices to real-values for actions
        :param actions: ndarray. list of indices for each action.
        :return: ndarray. the actual values for acceleration.
        """
        real_actions = np.array([-1.0, 1.0])
        return real_actions[actions]

    def unionize_actions(self, actions, state):
        """
        Combine the actions of each fuzzy state to a unified action.
        :param actions: ndarray. indices of chosen actions per fuzzy state.
        :param state: Box. the current state of the car.
        :return: float. the acceleration value.
        """
        alphas = self.extract_memberships(state)
        real = self.realize_actions(actions)
        return np.sum(real * alphas) / np.sum(alphas)

    def predict_v(self, state):
        """
        returns the estimation of value for a state based on the values of fuzzy states.
        :param state: Box. the current state of the car.
        :return: float. the estimated value for the given state.
        """
        alphas = self.extract_memberships(state)
        q_stars = np.max(self.q, axis=1)
        return np.sum(alphas * q_stars) / np.sum(alphas)

    def predict_q(self, actions, state):
        """
        returns the estimation of action-value for a state and a set of actions
        based on the values of fuzzy states.
        :param actions: ndarray. indices of chosen actions per fuzzy state.
        :param state: Box. the current state of the car.
        :return: float. the estimate action-value function for the given state and action.
        """
        alphas = self.extract_memberships(state)
        q_cross = [self.q[i, actions[i]] for i in range(64)]
        return np.sum(alphas * q_cross) / np.sum(alphas)

    def update_eligibility(self, actions, state):
        """
        updates the eligibility trace based on the taken actions and the current state.
        :param actions: ndarray. indices of chosen actions per fuzzy state.
        :param state: Box. the current state of the car.
        :return: None.
        """
        memberships = self.extract_memberships(state)
        total = np.sum(memberships)
        self.e *= 0.5 * self.gamma
        for i in range(64):
            self.e[i, actions[i]] += memberships[i] / total

    @staticmethod
    def extract_memberships(state):
        """
        Converts a continuous states to a series of fuzzy states. The feature space
        partitioning is visualized by fuzzy_feature_space.py
        :param state: Box. the current state of the car.
        :return: ndarray. The membership of the current in each fuzzy state
        """
        normalized = (state - np.array([-0.3, 0])) / np.array([1.8, 0.14])
        x, y = np.meshgrid(np.arange(-1.0, 1.2, 0.28), np.arange(-1.0, 1.2, 0.28))
        centers = np.vstack([x.ravel(), y.ravel()])
        values = -np.sum((normalized - centers.T) ** 2, axis=1) + 0.0625
        values[values < 0] = 0
        values /= 0.0625
        return values

    def train(self, episodes):
        """
        The implementation of Fuzzy Q-Learning
        :param episodes: int. number of episodes
        :return: None.
        """
        for _ in range(episodes):
            state = self.env.reset()
            while True:
                selected_actions = self.select_actions()
                action = self.unionize_actions(selected_actions, state)
                big_q = self.predict_q(selected_actions, state)
                next_state, reward, done, _ = self.env.step([action])
                big_v = self.predict_v(next_state)
                # want the car to get out as fast as possible
                if -0.1 < reward < 0.1:
                    reward = -0.1
                td_error = reward + (self.gamma * big_v) - big_q
                self.update_eligibility(selected_actions, state)
                for i in range(64):
                    self.q[i, selected_actions[i]] += (
                        self.alpha * td_error * self.e[i, selected_actions[i]]
                    )
                state = next_state
                if done:
                    break

    def _visualize_agent(self, state):
        self.env.render()
        actions = self.select_actions()
        action = self.unionize_actions(actions, state)
        next_state, _, done, _ = self.env.step([action])
        return next_state, done

    def show_off(self):
        """
        Uses current estimations of v and q to show off agent's abilities.
        :return: None
        """
        state = self.env.reset()
        while True:
            state, done = self._visualize_agent(state)
            if done:
                break
