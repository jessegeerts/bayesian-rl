import numpy as np
from scipy.stats import beta

from environment import SimpleMDP
from utils import softmax


class BayesQlearner(object):
    """Use bayesian Q learning to estimate a posterior over action values.

    Following Daw & Dayan, I use beta priors and Dearden's mixture update of the posterior.
    """

    def __init__(self, environment=SimpleMDP(), inv_temp=2):
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

    def train_one_episiode(self):
        self.env.reset()

        t = 0
        s = self.env.current_state

        while not self.env.is_terminal(self.env.current_state):
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


if __name__ == "__main__":

    agent = BayesQlearner()

    rvs = []
    for ep in range(20):
        agent.train_one_episiode()
        first_action_dist = agent.q_dists[0][1]
        rvs.append(first_action_dist)

    agent.q_dists