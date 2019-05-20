import numpy as np
import networkx as nx
import pandas as pd
import utils
import os


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
        self.state_indices = None

    def define_adjacency_graph(self):
        pass

    def get_adjacency_matrix(self):
        pass

    def create_graph(self):
        """Create networkx graph from adjacency matrix.
        """
        self.graph = nx.from_numpy_array(self.get_adjacency_matrix())

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


class SimpleMDP(Environment):
    """Very simple MDP with states on a linear track. Agent gets reward of 1 if it reaches last state.
    """
    def __init__(self, nr_states=3):
        Environment.__init__(self)
        self.nr_states = nr_states
        self.state_indices = np.arange(self.nr_states)
        self.nr_actions = 2
        self.actions = [0, 1]
        self.action_consequences = {0: -1, 1: +1}
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
            for action_key, consequence in self.action_consequences.items():
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
            return list(self.action_consequences)

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

    def get_current_state(self):
        """Return current state idx given current position.
        """
        return self.current_state

    def _fill_adjacency_matrix(self):
        self.adjacency_graph = np.zeros((self.nr_states, self.nr_states), dtype=np.int)
        for idx in self.state_indices:
            if (idx + 1) in self.state_indices:
                self.adjacency_graph[idx, idx + 1] = 1

    def get_adjacency_matrix(self):
        if self.adjacency_graph is None:
            self._fill_adjacency_matrix()
        return self.adjacency_graph


class GridWorld(Environment):
    """Grid world environment, where agent can move around and there is one (terminal) goal state. In this version,
    we read the grid environment from an MDP file. This way we can easily include walls, etc.
    """
    def __init__(self, env_name):
        Environment.__init__(self)
        if '.mdp' in env_name:
            env_path = env_name
        else:
            env_path = './mdps/{}.mdp'.format(env_name)

        self.matrix_MDP = None
        self.original_goal_x = None
        self.original_goal_y = None
        self.start_x = None
        self.start_y = None
        self.goal_x = None
        self.goal_y = None
        self.num_cols, self.num_rows = None, None
        self.nr_occupiable_states = None
        self.absorbing_states = []

        self.parse_string(env_path)
        self.name = os.path.splitext(os.path.split(env_path)[1])[0]
        self.curr_x = self.start_x
        self.curr_y = self.start_y

        self.nr_states = self.num_rows * self.num_cols
        self.state_indices = np.arange(self.nr_states)
        self.actions = ['up', 'right', 'down', 'left']
        self.action_idx = np.arange(len(self.actions))
        self.nr_actions = len(self.actions)

        self.reset_reward_func()

        self._fill_adjacency_matrix()
        self.create_graph()
        self.define_transition_probabilities()

    def reset_reward_func(self):
        if self.goal_x is not None and self.goal_y is not None:
            self.reward_func = np.zeros(self.nr_states)
            self.goal_state = self.get_state_idx(self.goal_x, self.goal_y)
            self.reward_func[self.goal_state] = 1

    def set_reward_func(self, rewards, set_goal=True):
        """Set the reward function of the environment.

        :param rewards: Vector of length self.nr_states containing the rewards for each states.
        :param set_goal: Reset the terminal state to be at the non-zero position of the reward vecctor.
        :return:
        """
        self.reward_func = rewards
        if set_goal:
            self.goal_state = np.where(rewards)[0][0]
            self.goal_x, self.goal_y = self.get_state_position(self.goal_state)

    def parse_string(self, path):
        """Parse the string describing the MDP. Everything is stored in a matrix such that -1 means wall and 0 means
        available state.

        :return:
        """
        description = self._read_file(path)
        self.nr_occupiable_states = description.count('.') + description.count('S') + description.count('G')
        data = description.split('\n')
        self.num_rows = int(data[0].split(',')[0])
        self.num_cols = int(data[0].split(',')[1])
        self.matrix_MDP = np.zeros((self.num_rows, self.num_cols))

        for i in range(len(data) - 1):
            for j in range(len(data[i + 1])):
                if data[i + 1][j] == 'X':  # Inaccessible (wall) state
                    self.matrix_MDP[i][j] = -1
                elif data[i + 1][j] == '.':
                    self.matrix_MDP[i][j] = 0
                elif data[i + 1][j] == 'S':  # Accessible state
                    self.matrix_MDP[i][j] = 0
                    self.start_x = i
                    self.start_y = j
                elif data[i + 1][j] == 'G':  # Goal state
                    self.matrix_MDP[i][j] = 0
                    self.original_goal_x = i
                    self.original_goal_y = j
                    self.goal_x = i
                    self.goal_y = j
                    self.absorbing_states.append([i, j])
                elif data[i + 1][j] == 'T':  # Absorbing but non-rewarding state.
                    self.matrix_MDP[i][j] = 0
                    self.absorbing_states.append([i, j])

    @staticmethod
    def _read_file(path):
        string = ''
        file = open(path, 'r')
        for line in file:
            string += line
        return string

    def get_state_idx(self, x, y):
        """Given coordinates, return the state index.
        """
        #idx = y + x * self.num_cols
        idx = x + y * self.num_rows
        return idx

    def get_state_position(self, idx):
        """Given the state index, return the x, y position.
        """
        #y = idx % self.num_cols
        #x = (idx - y) / self.num_cols
        x = idx % self.num_rows
        y = (idx - x) / self.num_rows

        return int(x), int(y)

    def get_next_state(self, origin, action):
        x, y = self.get_state_position(origin)
        if self.matrix_MDP[x][y] == -1:
            return origin

        if action == 'up' and y < self.num_rows - 1:
            next_x = x
            next_y = y + 1
        elif action == 'right' and x < self.num_cols - 1:
            next_x = x + 1
            next_y = y
        elif action == 'down' and y > 0:
            next_x = x
            next_y = y - 1
        elif action == 'left' and x > 0:
            next_x = x - 1
            next_y = y
        else:  # terminate
            next_state = self.nr_states  # FIXME: the weird termination problem is because of this. Look into this.
            return next_state
        if self.matrix_MDP[next_x][next_y] == -1:
            next_x = x
            next_y = y
        next_state = self.get_state_idx(next_x, next_y)
        return next_state

    def get_reward(self, origin, destination):
        if self.reward_func is None:
            return None
        reward = self.reward_func[destination]
        return reward

    def is_terminal(self, state_idx):
        return state_idx == self.goal_state

    def current_state_is_terminal(self):
        if self.curr_x == self.goal_x and self.curr_y == self.goal_y:
            return True
        else:
            for loc in self.absorbing_states:
                if loc[0] == self.curr_x and loc[1] == self.curr_y:
                    return True
            return False

    def get_next_state_and_reward(self, origin, action):
        # If current state is terminal absorbing state:
        if origin == self.nr_states:
            return origin, 0

        next_state = self.get_next_state(origin, action)
        reward = self.get_reward(origin, next_state)
        return next_state, reward

    def get_adjacency_matrix(self):
        if self.adjacency_graph is None:
            self._fill_adjacency_matrix()
        return self.adjacency_graph

    def _fill_adjacency_matrix(self):
        self.adjacency_graph = np.zeros((self.nr_states, self.nr_states), dtype=np.int)
        self.idx_matrix = np.zeros((self.num_rows, self.num_cols), dtype=np.int)

        for row in range(len(self.idx_matrix)):
            for col in range(len(self.idx_matrix[row])):
                self.idx_matrix[row][col] = row * self.num_cols + col

        for row in range(len(self.matrix_MDP)):
            for col in range(len(self.matrix_MDP[row])):
                if row != 0 and row != (self.num_rows - 1) and col != 0 and col != (self.num_cols - 1):
                    if self.matrix_MDP[row][col] != -1:
                        if self.matrix_MDP[row + 1][col] != -1:
                            self.adjacency_graph[self.idx_matrix[row][col]][self.idx_matrix[row + 1][col]] = 1
                        if self.matrix_MDP[row - 1][col] != -1:
                            self.adjacency_graph[self.idx_matrix[row][col]][self.idx_matrix[row - 1][col]] = 1
                        if self.matrix_MDP[row][col + 1] != -1:
                            self.adjacency_graph[self.idx_matrix[row][col]][self.idx_matrix[row][col + 1]] = 1
                        if self.matrix_MDP[row][col - 1] != -1:
                            self.adjacency_graph[self.idx_matrix[row][col]][self.idx_matrix[row][col - 1]] = 1

    def get_possible_actions(self, state_idx):
        if state_idx == self.goal_state:
            return []
        else:
            return self.actions #list(self.action_idx)

    def define_transition_probabilities(self):
        self.transition_probabilities = np.zeros([self.nr_states, self.nr_actions, self.nr_states])

        for origin in self.state_indices:
            x, y = self.get_state_position(origin)
            if self.matrix_MDP[x][y] != -1:
                if x != 0 and x != (self.num_rows - 1) and y != 0 and y != (self.num_cols - 1):
                    if self.matrix_MDP[x - 1][y] != -1:
                        self.transition_probabilities[origin, 0, self.get_state_idx(x - 1, y)] = 1
                    else:
                        self.transition_probabilities[origin, 0, origin] = 1
                    if self.matrix_MDP[x][y + 1] != -1:
                        self.transition_probabilities[origin, 1, self.get_state_idx(x, y + 1)] = 1
                    else:
                        self.transition_probabilities[origin, 1, origin] = 1
                    if self.matrix_MDP[x + 1][y] != -1:
                        self.transition_probabilities[origin, 2, self.get_state_idx(x + 1, y)] = 1
                    else:
                        self.transition_probabilities[origin, 2, origin] = 1
                    if self.matrix_MDP[x][y - 1] != -1:
                        self.transition_probabilities[origin, 3, self.get_state_idx(x, y - 1)] = 1
                    else:
                        self.transition_probabilities[origin, 3, origin] = 1

    def reset(self):
        """Reset agent to start position.
        """
        self.curr_x = self.start_x
        self.curr_y = self.start_y

    def get_current_state(self):
        """Return current state idx given current position.
        """
        current_state_idx = self.get_state_idx(self.curr_x, self.curr_y)
        return current_state_idx

    def act(self, action):
        """

        :param action:
        :return:
        """
        current_state = self.get_current_state()
        if self.reward_func is None and self.is_terminal(current_state):
            return 0
        else:
            next_state, reward = self.get_next_state_and_reward(current_state, action)
            next_x, next_y = self.get_state_position(next_state)
            self.curr_x = next_x
            self.curr_y = next_y
            return next_state, reward


class TransitionRevaluation(Environment):
    """This class simulates the transition revaluation experiment designed by Momennejad et al. (2017).
    """
    def __init__(self):
        Environment.__init__(self)
        self.nr_states = 7
        self.state_indices = list(range(self.nr_states))
        self.nr_actions = None
        self.actions = None

        self.transitions = {
            0: None,  # Zero is the terminal state.
            1: 3,
            2: 4,
            3: 5,
            4: 6,
            5: 0,
            6: 0
        }

        self.reward_function = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 10,
            6: 1
        }

        self.transition_probabilities = self.define_transition_probabilities()

        self.possible_start_states = [1, 2]
        self.current_state = np.random.choice(self.possible_start_states)

    def reset(self):
        self.current_state = np.random.choice(self.possible_start_states)

    def define_transition_probabilities(self):
        transition_probabilities = np.zeros([self.nr_states, self.nr_states])
        for predecessor in self.state_indices:
            if self.is_terminal(predecessor):
                transition_probabilities[predecessor, :] = 0
                continue

            successor = self.transitions[predecessor]

            transition_probabilities[predecessor, successor] = 1
        return transition_probabilities

    def is_terminal(self, state_idx):
        return True if state_idx == 0 else False

    def act(self, action=None):
        """Gets next state given previous state.
        """
        next_state, reward = self.get_next_state_and_reward(self.current_state)
        self.current_state = next_state
        return next_state, reward

    def get_next_state_and_reward(self, current_state):
        next_state = self.get_next_state(current_state)
        reward = self.get_reward(next_state)
        return next_state, reward

    def get_next_state(self, current_state):
        return self.transitions[current_state]

    def get_reward(self, state):
        return self.reward_function[state]

    def get_current_state(self):
        return self.current_state

    def set_relearning_phase(self):
        self.possible_start_states = [3, 4]

        self.transitions = {
            0: None,  # Zero is the terminal state.
            1: 3,
            2: 4,
            3: 6,
            4: 5,
            5: 0,
            6: 0
        }



if __name__ == '__main__':
    from agent import LinearKalmanSRTD

    en = TransitionRevaluation()

    ag = LinearKalmanSRTD(en)
    ag.train_one_episode()