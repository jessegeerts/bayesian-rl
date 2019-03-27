import numpy as np
import networkx as nx
import pandas as pd
import utils


class Environment(object):
    """Parent class for RL environments holding some general methods.
    """
    def __init__(self):
        self.nr_states = None
        self.nr_actions = None
        self.adjacency_graph = None
        self.goal_state = None
        self.reward_func = None
        self.graph = None
        self.n_features = None
        self.rf = None
        self.transition_probabilities = None
        self.terminal_state = None

    def define_adjacency_graph(self):
        pass

    def create_graph(self):
        """Create networkx graph from adjacency matrix.
        """
        self.graph = nx.from_numpy_array(self.adjacency_graph)

    def show_graph(self, map_variable=None, layout=None, node_size=1500, **kwargs):
        """Plot graph showing possible state transitions.

        :param node_size:
        :param map_variable: Continuous variable that can be mapped on the node colours.
        :param layout:
        :param kwargs: Any other drawing parameters accepted. See nx.draw docs.
        :return:
        """
        if layout is None:
            layout = nx.spring_layout(self.graph)
        if map_variable is not None:
            categories = pd.Categorical(map_variable)
            node_color = categories
        else:
            node_color = 'b'
        nx.draw(self.graph, with_labels=True, pos=layout, node_color=node_color, node_size=node_size, **kwargs)

    def set_reward_location(self, state_idx, action_idx):
        self.goal_state = state_idx
        action_destination = self.transition_probabilities[state_idx, action_idx]
        self.reward_func = np.zeros([self.nr_states, self.nr_actions, self.nr_states])
        self.reward_func[state_idx, action_idx] = action_destination

    def is_terminal(self, state_idx):
        if not self.get_possible_actions(state_idx):
            return True
        else:
            return False

    def get_destination_state(self, current_state, current_action):
        transition_probabilities = self.transition_probabilities[current_state, current_action]
        return np.flatnonzero(transition_probabilities)

    def get_degree_mat(self):
        degree_mat = np.eye(self.nr_states)
        for state, degree in self.graph.degree:
            degree_mat[state, state] = degree
        return degree_mat

    def get_laplacian(self):
        return self.get_degree_mat() - self.adjacency_graph

    def get_normalised_laplacian(self):
        """Return the normalised laplacian.
        """
        D = self.get_degree_mat()
        L = self.get_laplacian()  # TODO: check diff with non normalised laplacian. check adverserial examples
        exp_D = utils.exponentiate(D, -.5)
        return exp_D.dot(L).dot(exp_D)

    def compute_laplacian(self, normalization_method=None):
        """Compute the Laplacian.

        :param normalization_method: Choose None for unnormalized, 'rw' for RW normalized or 'sym' for symmetric.
        :return:
        """
        if normalization_method not in [None, 'rw', 'sym']:
            raise ValueError('Not a valid normalisation method. See help(compute_laplacian) for more info.')

        D = self.get_degree_mat()
        L = D - self.adjacency_graph

        if normalization_method is None:
            return L
        elif normalization_method == 'sym':
            exp_D = utils.exponentiate(D, -.5)
            return exp_D.dot(L).dot(exp_D)
        elif normalization_method == 'rw':
            exp_D = utils.exponentiate(D, -1)
            return exp_D.dot(L)

    def get_possible_actions(self, state_idx):
        pass

    def get_adjacent_states(self, state_idx):
        pass

    def compute_feature_response(self):
        pass


class SimpleMDP(Environment):
    """Very simple MDP with three states. Agent gets reward of 1 if it reaches third state.
    """
    def __init__(self, nr_states=3):
        Environment.__init__(self)
        self.nr_states = nr_states
        self.nr_actions = 2
        self.state_indices = np.arange(self.nr_states)
        self.nr_actions = 2
        self.actions = {0: -1, 1: +1}
        self.terminal_states = [self.nr_states - 1]

        self.transition_probabilities = self.define_transition_probabilities()
        self.reward_func = np.zeros((self.nr_states, self.nr_actions))
        self.reward_func[self.nr_states-2, 1] = 1
        self.current_state = 0

    def reset(self):
        self.current_state = 0

    def define_transition_probabilities(self):
        transition_probabilities = np.zeros([self.nr_states, self.nr_actions, self.nr_states])
        for predecessor in self.state_indices:
            if self.is_terminal(predecessor):
                transition_probabilities[predecessor, :, :] = 0
                continue
            for action_key, consequence in self.actions.items():
                successor = int(predecessor + consequence)
                if successor not in self.state_indices:
                    transition_probabilities[predecessor, action_key, predecessor] = 1  # stay in current state
                else:
                    transition_probabilities[predecessor, action_key, successor] = 1
        return transition_probabilities

    def get_possible_actions(self, state_idx):
        if state_idx in self.terminal_states:
            return []
        else:
            return list(self.actions)

    def define_adjacency_graph(self):
        transitions_under_random_policy = self.transition_probabilities.sum(axis=1)
        adjacency_graph = transitions_under_random_policy != 0
        return adjacency_graph.astype('int')

    def get_transition_matrix(self, policy):
        transition_matrix = np.zeros([self.nr_states, self.nr_states])
        for state in self.state_indices:
            if self.is_terminal(state):
                continue
            for action in range(self.nr_actions):
                transition_matrix[state] += self.transition_probabilities[state, action] * policy[state][action]
        return transition_matrix

    def get_successor_representation(self, policy, gamma=.95):
        transition_matrix = self.get_transition_matrix(policy)
        m = np.linalg.inv(np.eye(self.nr_states) - gamma * transition_matrix)
        return m

    def get_next_state(self, current_state, action):
        next_state = np.flatnonzero(self.transition_probabilities[current_state, action])[0]
        return next_state

    def get_reward(self, current_state, action):
        return self.reward_func[current_state, action]

    def get_next_state_and_reward(self, current_state, action):
        # If current state is terminal absorbing state:
        if self.is_terminal(current_state):
            return current_state, 0

        next_state = self.get_next_state(current_state, action)
        reward = self.get_reward(current_state, action)
        return next_state, reward

    def act(self, action):
        next_state, reward = self.get_next_state_and_reward(self.current_state, action)
        self.current_state = next_state
        return next_state, reward

