import numpy as np
from scipy.stats import beta
from scipy.linalg import ldl
from itertools import product

from environment import SimpleMDP, GridWorld, Environment
from utils import softmax, dotproduct_var, product_var


class BayesQlearner(object):
    """Use bayesian Q learning (Dearden) to estimate a posterior over action values.

    Following Daw & Dayan, I use beta priors and Dearden's mixture update of the posterior.
    """

    def __init__(self, environment=Environment(), inv_temp=2):
        self.inv_temp = inv_temp
        self.env = environment
        self.actions = np.arange(self.env.nr_actions)
        self.nr_params = 2  # alpha and beta, the parameters determining the beta distribution
        self.q_params = np.empty((self.env.nr_states, self.env.nr_actions, self.nr_params))
        self.prior_a = .1
        self.prior_b = 1

        self.q_dists = []
        for s in range(self.env.nr_states):
            actions = self.env.get_possible_actions(s)
            qd = []
            if not actions:
                self.q_dists.append(beta(self.prior_a, self.prior_b))
            else:
                for a in actions:
                    qd.append(beta(self.prior_a, self.prior_b))
            self.q_dists.append(qd)

    def train_one_episode(self):
        self.env.reset()

        t = 0
        s = self.env.get_current_state()

        while not self.env.is_terminal(self.env.get_current_state()):
            q_value_means = [q.mean() for q in self.q_dists[s]]
            action_probabilities = softmax(q_value_means, beta=self.inv_temp)
            a = np.random.choice(self.actions, p=action_probabilities)
            next_state, reward = self.env.act(a)

            if self.env.is_terminal(next_state):
                # Update terminal state reward distribution:
                self.q_dists[next_state] = self.update_reward_distribution(next_state, reward)

            # update value distribution of predecessor state action pair
            self.q_dists[s][a] = self.update_qvalue_distribution(s, a, next_state)

            s = next_state
            t += 1
        return t

    def update_qvalue_distribution(self, s, a, next_state):
        """Update the Q value distribution for state-action pair (s,a).

        The successor state's value is used as bootstrapped sample of the predecessor state's mean value.
        I compute the mean and variance of the posterior distribution, and then estimate the updated alpha
        and beta parameters (see Daw & Dayan, 2005).

        :param s: Predecessor state.
        :param a: Current action.
        :param next_state: Successor state.
        :return:
        """
        post_mean = self.update_posterior_mean(s, a, next_state)
        post_var = self.update_posterior_var(s, a, next_state, post_mean)
        pre_alpha, pre_beta = self.est_beta_params(post_mean, post_var)
        posterior = beta(pre_alpha, pre_beta)
        return posterior

    def update_posterior_mean(self, s, a, next_state):
        """Compute mean of posterior Q value distribution.

        :param s:
        :param a:
        :param next_state:
        :return:
        """
        if self.env.is_terminal(next_state):
            next_state_dist = self.q_dists[next_state]
        else:
            a_prime = np.argmax([q.mean() for q in self.q_dists[next_state]])
            next_state_dist = self.q_dists[next_state][a_prime]

        alpha_s = self.get_alpha(self.q_dists[s][a])
        beta_s = self.get_beta(self.q_dists[s][a])
        post_mean = (alpha_s + next_state_dist.moment(1)) / (alpha_s + beta_s + 1)
        return post_mean

    def update_posterior_var(self, s, a, next_state, posterior_mean):
        """Compute variance of posterior Q value distribution.

        :param s:
        :param a:
        :param next_state:
        :param posterior_mean:
        :return:
        """
        if self.env.is_terminal(next_state):
            next_state_dist = self.q_dists[next_state]
        else:
            a_prime = np.argmax([q.mean() for q in self.q_dists[next_state]])
            next_state_dist = self.q_dists[next_state][a_prime]
        alpha_s = self.get_alpha(self.q_dists[s][a])
        beta_s = self.get_beta(self.q_dists[s][a])

        post_var = 1 / ((alpha_s + beta_s + 2) * (alpha_s + beta_s + 1)) * (
                alpha_s ** 2 + alpha_s + next_state_dist.moment(2) + (2 * alpha_s + 1) *
                next_state_dist.moment(1)) - posterior_mean ** 2
        return post_var

    @staticmethod
    def get_alpha(beta_rv):
        return beta_rv.args[0]

    @staticmethod
    def get_beta(beta_rv):
        return beta_rv.args[1]

    def update_reward_distribution(self, terminal_state_idx, reward):
        """Compute posterior reward distribution in terminal state.

        :param terminal_state_idx:
        :param reward:
        :return:
        """
        new_alpha = self.q_dists[terminal_state_idx].args[0] + reward
        new_beta = self.q_dists[terminal_state_idx].args[1] + (1 - reward)
        posterior = beta(new_alpha, new_beta)
        return posterior

    @staticmethod
    def est_beta_params(mu, var):
        """Estimate the alpha and beta parameters of the beta distribution given its mean and variance.

        :param mu:
        :param var:
        :return:
        """
        alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
        beta = alpha * (1 / mu - 1)
        return alpha, beta


class KTDV(object):
    """Implementation of the Kalman TD value approximation algorithm with linear function approximation (Geist &
    Pietquin, 2012).

    We ignore control for now. The agent follows a random policy.
    """
    def __init__(self, environment=Environment(), gamma=.9, inv_temp=2):
        self.env = environment
        self.actions = self.env.actions

        # Parameters
        self.transition_noise = .005 * np.eye(self.env.nr_states)
        self.gamma = gamma
        self.inv_temp = inv_temp  # exploration parameter
        self.observation_noise_variance = 1

        # Initialise priors
        self.prior_theta = np.zeros(self.env.nr_states)
        self.prior_covariance = np.eye(self.env.nr_states)

        self.theta = self.prior_theta
        self.covariance = self.prior_covariance

    def train_one_episode(self, random_policy=False, fixed_policy=False):
        self.env.reset()

        t = 0
        s = self.env.get_current_state()
        features = self.get_feature_representation(s)

        results = {}

        while not self.env.is_terminal(self.env.get_current_state()) and t < 1000:
            # Observe transition and reward;
            if random_policy:
                a = np.random.choice(self.actions)
            elif fixed_policy:
                a = 1
            else:
                a = self.select_action()

            next_state, reward = self.env.act(a)

            next_features = self.get_feature_representation(next_state)
            H = features - self.gamma * next_features  # Temporal difference features

            # Prediction step;
            a_priori_covariance = self.covariance + self.transition_noise

            # Compute statistics of interest;
            V = np.dot(features, self.theta)
            r_hat = np.dot(H, self.theta)
            delta_t = reward - r_hat  # the "prediction error". Actually expected value of the prediction error
            residual_cov = np.dot(H, np.matmul(a_priori_covariance, H)) + self.observation_noise_variance

            # Correction step;
            kalman_gain = np.matmul(a_priori_covariance, H) * residual_cov**-1
            self.theta = self.theta + kalman_gain * delta_t  # weight update
            self.covariance = a_priori_covariance - np.outer(kalman_gain, residual_cov*kalman_gain)
            #self.covariance = new_cov - np.matmul(np.outer(kalman_gain, H), new_cov)
            #TODO: There's a discrepancy here between Gershman and Pietquin. resolve this. Mathematically equivalent?

            # Store results
            results[t] = {'weights': self.theta,
                          'cov': self.covariance,
                          'K': kalman_gain,
                          'dt': delta_t,
                          'r': reward,
                          'state': s,
                          'rhat': r_hat,
                          'V': V}

            s = next_state
            features = self.get_feature_representation(s)
            t += 1

        return results

    def select_action(self):
        Q = []
        for idx, a in enumerate(self.actions):
            s_prime = self.env.get_destination_state(self.env.get_current_state(), idx)[0]
            features = self.get_feature_representation(s_prime)
            V = np.dot(features, self.theta)
            if self.env.is_terminal(s_prime):
                V = 1
            Q.append(V)

        action = np.random.choice(self.actions, p=softmax(Q, beta=self.inv_temp))
        return action

    def get_feature_representation(self, state_idx):
        """Get one-hot feature representation from state index.
        """
        if self.env.is_terminal(state_idx):
            return np.zeros(self.env.nr_states)
        else:
            return np.eye(self.env.nr_states)[state_idx]


class XKTDV(object):
    """Extended KTD-V (algorithm 5 in Geist & Pietquin). Uses coloured noise model.
    """
    def __init__(self, environment=Environment(), transition_noise=.005, gamma=.9, inv_temp=2, kappa=1.):
        self.env = environment

        # Parameters
        self.n = self.env.nr_states
        self.gamma = gamma
        self.inv_temp = inv_temp  # exploration parameter
        self.observation_noise_variance = 1
        self.kappa = kappa

        # for coloured noise model:
        self.F = np.zeros((self.n + 2, self.n + 2))
        self.F[:-2, :-2] = np.eye(self.n)
        self.F[-2:, -2:] = np.array([[0, 0],
                                     [1, 0]])

        self.sigma = 1
        self.Cu = self.sigma * np.array([[1, -self.gamma], [-self.gamma, self.gamma**2]])
        self.transition_noise = np.zeros((self.n + 2, self.n + 2))
        self.transition_noise[:-2, :-2] = transition_noise * np.eye(self.n)
        self.transition_noise[-2:, -2:] = self.Cu

        # Initialise priors
        self.prior_x = np.zeros(self.n + 2)  # Note: x contains parameters theta appended with vectorial AR noise
        self.prior_covariance = np.eye(self.n + 2) * .01

        self.x = self.prior_x
        self.covariance = self.prior_covariance

    def train_one_episode(self):
        self.env.reset()

        t = 0
        s = self.env.get_current_state()
        features = self.get_feature_representation(s)

        results = {}

        while not self.env.is_terminal(self.env.get_current_state()) and t < 1000:
            # observe transition and reward
            a = 1

            next_state, reward = self.env.act(a)
            next_features = self.get_feature_representation(next_state)
            H = features - self.gamma * next_features  # temporal difference features

            # Prediction step
            self.x = self.F @ self.x
            self.covariance = self.F @ self.covariance @ self.F.T + self.transition_noise

            # Sigma points computation
            X, weights = self.sample_sigma_points()
            Y = np.array([np.dot(H, X[j, :-2]) for j in range(X.shape[0])])

            r_hat = np.dot(weights, Y)
            Cxr = np.sum([weights[j] * (X[j] - self.x) * (Y[j] - r_hat) for j in range(X.shape[0])], axis=0)
            Cr = np.sum([weights[j] * (Y[j] - r_hat) * (Y[j] - r_hat) for j in range(X.shape[0])], axis=0)

            # Correction step
            K = Cxr / Cr
            self.x += K * (reward - r_hat)
            self.covariance -= np.outer(K,  Cr * K)

            # Store results
            results[t] = {'x': self.x,
                          'cov': self.covariance,
                          'K': K,
                          'dt': reward - r_hat,
                          'r': reward,
                          'state': s}

            s = next_state
            features = self.get_feature_representation(s)
            t += 1

        return results

    def sample_sigma_points(self):
        n = len(self.x)
        X = np.empty((2 * n + 1, n))
        X[:, :] = self.x[None, :]  # fill array with m for each sample
        try:
            C = np.linalg.cholesky((self.kappa + n) * self.covariance)
        except:
            print('Cholesky decomposition did not work')
            C, d, perm = ldl((self.kappa + n) * self.covariance)

        for j in range(n):
            X[j + 1, :] += C[:, j]
            X[j + n + 1, :] -= C[:, j]

        weights = np.ones(2 * n + 1) * (1. / (2 * (self.kappa + n)))
        weights[0] = (self.kappa / (self.kappa + n))
        return X, weights

    def get_feature_representation(self, state_idx):
        """Get one-hot feature representation from state index.
        """
        if self.env.is_terminal(state_idx):
            return np.zeros(self.env.nr_states)
        else:
            return np.eye(self.env.nr_states)[state_idx]


class KalmanSR(object):
    """Estimate the successor representation (Dayan, 1993) using Kalman TD. The policy is deterministic,
    so the only problem solved here is prediction.
    """
    def __init__(self, environment=Environment(), gamma=.9, inv_temp=2):
        self.env = environment
        self.actions = self.env.actions

        # Parameters
        self.transition_noise = .005 * np.eye(self.env.nr_states)
        self.gamma = gamma
        self.inv_temp = inv_temp  # exploration parameter
        self.observation_noise_variance = 1

        # Initialise priors
        self.prior_w = np.zeros(self.env.nr_states)
        self.prior_M = np.eye(self.env.nr_states)
        self.prior_covariance = np.eye(self.env.nr_states)

        self.M = self.prior_M
        self.w = self.prior_w
        self.covariance = self.prior_covariance

    def train_one_episode(self):
        self.env.reset()

        t = 0
        s = self.env.get_current_state()
        features = self.get_feature_representation(s)

        results = {}

        while not self.env.is_terminal(self.env.get_current_state()) and t < 1000:
            # Observe transition and reward;
            a = 1

            next_state, reward = self.env.act(a)

            next_features = self.get_feature_representation(next_state)
            H = features - self.gamma * next_features  # Temporal difference features

            # Prediction step;
            a_priori_covariance = self.covariance + self.transition_noise

            # Compute statistics of interest;
            phi_hat = np.matmul(self.M.T, H)
            r_hat = np.dot(self.w, features)
            delta_t = features - phi_hat
            rpe = reward - r_hat
            residual_cov = np.dot(H, np.matmul(a_priori_covariance, H)) + self.observation_noise_variance

            # Correction step;
            kalman_gain = np.matmul(a_priori_covariance, H) * residual_cov**-1

            delta_M = np.outer(kalman_gain, delta_t)
            self.M += delta_M
            self.w = self.w + kalman_gain * rpe
            self.covariance = a_priori_covariance - np.outer(kalman_gain, residual_cov*kalman_gain)

            V = np.matmul(self.M, self.w)
            #V_variance = np.array([dotproduct_var(self.M[i, :-1],
            #                                      self.w[:-1],
            #                                      np.diag(self.covariance)[:-1],
            #                                      np.diag(self.covariance[:-1])) for i in range(self.env.nr_states)])

            V_variance = np.array([product_var(self.M[i, -2], self.w[-2],
                                      np.diag(self.covariance)[i], np.diag(self.covariance)[i])
                          for i in range(self.env.nr_states)])

            # Store results
            results[t] = {'SR': self.M,
                          'cov': self.covariance,
                          'K': kalman_gain,
                          'dt': delta_t,
                          'r': reward,
                          'state': s,
                          'V': V,
                          'V_var': V_variance,
                          'w': self.w}

            # TODO: after lunch, check this works but also compute variance over V
            s = next_state
            features = self.get_feature_representation(s)
            t += 1

        return results

    def get_feature_representation(self, state_idx):
        """Get one-hot feature representation from state index.
        """
        if self.env.is_terminal(state_idx):
            return np.zeros(self.env.nr_states)
        else:
            return np.eye(self.env.nr_states)[state_idx]


class UnscentedKalmanSRTD(object):
    """Estimate the successor representation (Dayan, 1993) using Kalman TD. The policy is deterministic,
    so the only problem solved here is prediction. We use the unscented transform, so that any time of function
    approximation can be applied.
    """
    def __init__(self, environment=Environment(), gamma=.9, kappa=1., inv_temp=2):
        self.env = environment
        self.actions = self.env.actions

        # Parameters
        self.transition_noise = .005 * np.eye(self.env.nr_states**2)
        self.gamma = gamma
        self.kappa = kappa
        self.inv_temp = inv_temp  # exploration parameter
        self.observation_noise_variance = np.eye(self.env.nr_states)

        # Initialise priors
        self.prior_M = np.eye(self.env.nr_states).flatten()
        self.prior_covariance = np.eye(self.env.nr_states**2) * .1

        self.M = self.prior_M
        self.covariance = self.prior_covariance

    def train_one_episode(self):
        self.env.reset()

        t = 0
        s = self.env.get_current_state()
        features = self.get_feature_representation(s)

        results = {}

        while not self.env.is_terminal(self.env.get_current_state()) and t < 1000:
            # Observe transition and reward;
            a = 1

            next_state, reward = self.env.act(a)

            next_features = self.get_feature_representation(next_state)
            H = features - self.gamma * next_features  # Temporal difference features
            feature_block_matrix = np.kron(H, np.eye(self.env.nr_states)).T

            # Prediction step;
            a_priori_covariance = self.covariance + self.transition_noise

            # Compute sigma points
            X, weights = self.sample_sigma_points()
            Y = np.matmul(X, feature_block_matrix)

            # Compute statistics of interest;
            phi_hat = np.multiply(Y, weights[:, np.newaxis]).sum(axis=0)
            param_error_cov = np.sum([weights[j] * np.outer((X[j] - self.M), (Y[j] - phi_hat))
                                      for j in range(len(weights))], axis=0)  # TODO: rewrite as matmul
            residual_cov = np.maximum(np.sum([weights[j] * np.outer((Y[j] - phi_hat), (Y[j] - phi_hat))
                                              for j in range(len(weights))], axis=0), 10e-5)
            delta_t = features - phi_hat

            # Correction step;
            kalman_gain = np.matmul(param_error_cov, np.linalg.inv(residual_cov))

            delta_M = np.matmul(kalman_gain, delta_t)
            self.M += delta_M
            self.covariance = a_priori_covariance - np.matmul(np.matmul(kalman_gain, residual_cov), kalman_gain.T)

            # Store results
            results[t] = {'SR': self.M,
                          'cov': self.covariance,
                          'K': kalman_gain,
                          'dt': delta_t,
                          'r': reward,
                          'state': s}

            s = next_state
            features = self.get_feature_representation(s)
            t += 1

        return results

    def get_feature_representation(self, state_idx):
        """Get one-hot feature representation from state index.
        """
        if self.env.is_terminal(state_idx):
            return np.zeros(self.env.nr_states)
        else:
            return np.eye(self.env.nr_states)[state_idx]

    def sample_sigma_points(self):
        n = len(self.M)
        X = np.empty((2 * n + 1, n))
        X[:, :] = self.M[None, :]  # fill array with m for each sample
        try:
            C = np.linalg.cholesky((self.kappa + n) * self.covariance)
        except:
            C, d, perm = ldl((self.kappa + n) * self.covariance)

        for j in range(n):
            X[j + 1, :] += C[:, j]
            X[j + n + 1, :] -= C[:, j]

        weights = np.ones(2 * n + 1) * (1. / (2 * (self.kappa + n)))
        weights[0] = (self.kappa / (self.kappa + n))
        return X, weights


class LinearKalmanSRTD(object):
    """Estimate the successor representation (Dayan, 1993) using Kalman TD. The policy is deterministic,
    so the only problem solved here is prediction.
    """
    def __init__(self, environment=Environment(), gamma=.9, kappa=1., inv_temp=2):
        self.env = environment
        self.actions = self.env.actions

        self.control = True if self.env.actions is not None else False

        # Parameters
        self.n = self.env.nr_states
        self.transition_noise = .0001 * np.eye(self.n ** 2)
        self.r_transition_noise = .01 * np.eye(self.n)
        self.gamma = gamma
        self.kappa = kappa
        self.inv_temp = inv_temp  # exploration parameter
        self.observation_noise_variance = np.eye(self.n)
        self.r_observation_noise_variance = 1

        # Initialise priors
        self.prior_w = np.zeros(self.env.nr_states)

        self.prior_M = np.eye(self.n).flatten()
        self.prior_covariance = np.eye(self.n ** 2) * .1
        self.prior_reward_covariance = np.eye(self.n)

        self.M = self.prior_M
        self.w = self.prior_w
        self.covariance = self.prior_covariance
        self.reward_covariance = self.prior_reward_covariance

        self.m_labels = ['M{}-{}'.format(i, j) for i, j in product(list(range(self.n)), list(range(self.n)))]

    def train_one_episode(self):
        self.env.reset()

        t = 0
        s = self.env.get_current_state()
        features = self.get_feature_representation(s)

        results = {}

        while not self.env.is_terminal(self.env.get_current_state()) and t < 1000:
            # Observe transition and reward;
            a = 1  # = np.random.choice(self.env.actions)

            next_state, reward = self.env.act(a)

            next_features = self.get_feature_representation(next_state)
            H = features - self.gamma * next_features  # Temporal difference features
            feature_block_matrix = np.kron(H, np.eye(self.n)).T

            # Prediction step;
            a_priori_covariance = self.covariance + self.transition_noise
            a_priori_r_covariance = self.reward_covariance + self.r_transition_noise

            # Compute statistics of interest;
            phi_hat = np.matmul(feature_block_matrix.T, self.M)
            r_hat = np.dot(self.w, features)
            parameter_error_cov = np.matmul(a_priori_covariance, feature_block_matrix)
            residual_cov = np.matmul(np.matmul(feature_block_matrix.T, a_priori_covariance),
                                     feature_block_matrix) + self.observation_noise_variance
            # For RW:
            residual_cov_rw = features @ a_priori_r_covariance @ features.T + self.r_observation_noise_variance

            delta_t = features - phi_hat
            rpe = reward - r_hat

            # Correction step;
            kalman_gain = np.matmul(parameter_error_cov, np.linalg.inv(residual_cov))
            r_kalman_gain = a_priori_r_covariance @ features.T / residual_cov_rw

            delta_M = np.matmul(kalman_gain, delta_t)
            self.M += delta_M
            self.w += r_kalman_gain * rpe
            self.covariance = a_priori_covariance - np.matmul(np.matmul(kalman_gain, residual_cov), kalman_gain.T)
            self.reward_covariance = self.reward_covariance - np.outer(r_kalman_gain, features) @ a_priori_r_covariance

            # Store results
            results[t] = {'SR': self.M,
                          'cov': self.covariance,
                          'K': kalman_gain,
                          'dt': delta_t,
                          'r': reward,
                          'state': s}

            s = next_state
            features = self.get_feature_representation(s)
            t += 1

        return results

    def get_feature_representation(self, state_idx):
        """Get one-hot feature representation from state index.
        """
        if self.env.is_terminal(state_idx):
            return np.zeros(self.env.nr_states)
        else:
            return np.eye(self.env.nr_states)[state_idx]

    def get_covariance(self, m_idx_1, m_idx_2):
        """Returns the estimated covariance between two parameters.

        :param m_idx_1: A 2D tuple or list containing the index in the SR matrix (one index for each dimension).
        :param m_idx_2: A 2D tuple or list containing the index in the SR matrix (one index for each dimension).
        :return:
        """
        flat_idx_1 = np.ravel_multi_index(m_idx_1, (self.n, self.n))
        flat_idx_2 = np.ravel_multi_index(m_idx_2, (self.n, self.n))
        return self.covariance[flat_idx_1, flat_idx_2]



if __name__ == "__main__":
    from environment import TransitionRevaluation

    alg = 5

    if alg == 0:
        agent = KTDV(environment=GridWorld('./mdps/10x10.mdp'))

        all_results = {}
        for ep in range(200):
            results = agent.train_one_episode(random_policy=False)
            all_results[ep] = results

    print('')

    if alg == 1:

        agent = BayesQlearner()

        rvs = []
        for ep in range(20):
            agent.train_one_episode()
            first_action_dist = agent.q_dists[0][1]
            rvs.append(first_action_dist)


    if alg == 2:
        agent = KTDV(environment=SimpleMDP(nr_states=3))

        all_results = {}
        for ep in range(50):
            results = agent.train_one_episode(fixed_policy=True)
            all_results[ep] = results


    if alg == 3:
        agent = KalmanSR(environment=SimpleMDP(nr_states=5))

        all_results = {}
        for ep in range(50):
            results = agent.train_one_episode()
            all_results[ep] = results

    if alg == 4:
        agent = UnscentedKalmanSRTD(environment=SimpleMDP(nr_states=5))

        all_results = {}
        for ep in range(50):
            results = agent.train_one_episode()
            all_results[ep] = results

    if alg == 5:
        agent = LinearKalmanSRTD(environment=TransitionRevaluation())

        all_results = {}
        for ep in range(50):
            results = agent.train_one_episode()
            all_results[ep] = results

        agent.env.set_relearning_phase()

        for i in range(20):
            agent.train_one_episode()


    print('')

