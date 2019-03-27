import numpy as np
from scipy.stats import beta

from environment import SimpleMDP
from utils import softmax


class BayesQlearner(object):
    """Use bayesian Q learning to estimate action values with confidence.

    At some point, we're gonna have to estimate the beta parameters given new mean and variance.

    R code for doing so:

    estBetaParams <- function(mu, var) {
        alpha <- ((1 - mu) / var - 1 / mu) * mu ^ 2
        beta <- alpha * (1 / mu - 1)
        return(params = list(alpha = alpha, beta = beta))
    }

    """

    def __init__(self, environment=SimpleMDP(), inv_temp=2):
        self.inv_temp = inv_temp
        self.env = environment
        self.actions = np.arange(self.env.nr_actions)
        self.nr_params = 2  # alpha and beta, the parameters determining the beta distribution
        self.q_params = np.empty((self.env.nr_states, self.env.nr_actions, self.nr_params))
        self.prior_a = 1
        self.prior_b = 1
        #self.q_dists = [[beta(self.prior_a, self.prior_b)
        #                 for a in range(self.env.nr_actions)]
        #                for s in range(self.env.nr_states)]

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
                # Update reward distribution of terminal state
                r_alpha = self.q_dists[next_state].args[0] + reward
                r_beta = self.q_dists[next_state].args[1] + (1 - reward)
                self.q_dists[next_state] = beta(r_alpha, r_beta)
                # update value distribution of predecessor state action pair
                post_mean = (self.q_dists[s][a].args[0] + self.q_dists[next_state].moment(1)) / \
                            (sum(self.q_dists[s][a].args) + 1)

                pre_a = self.q_dists[s][a].args[0]
                pre_b = self.q_dists[s][a].args[1]
                post_var = 1 / ((pre_a + pre_b + 2) * (pre_a + pre_b + 1)) * (
                            pre_a ** 2 + pre_a + self.q_dists[next_state].moment(2) + (2 * pre_a + 1) *
                            self.q_dists[next_state].moment(1)) - post_mean ** 2
                pre_alpha, pre_beta = self.est_beta_params(post_mean, post_var)
                self.q_dists[s][a] = beta(pre_alpha, pre_beta)
            else:
                # update value distribution of predecessor state action pair
                a_prime = np.argmax([q.mean() for q in self.q_dists[next_state]])
                post_mean = (self.q_dists[s][a].args[0] + self.q_dists[next_state][a_prime].moment(1)) / \
                            (sum(self.q_dists[s][a].args) + 1)

                pre_a = self.q_dists[s][a].args[0]
                pre_b = self.q_dists[s][a].args[1]
                post_var = 1 / ((pre_a + pre_b + 2) * (pre_a + pre_b + 1)) * (
                            pre_a ** 2 + pre_a + self.q_dists[next_state][a_prime].moment(2) + (2 * pre_a + 1) *
                            self.q_dists[next_state][a_prime].moment(1)) - post_mean ** 2
                pre_alpha, pre_beta = self.est_beta_params(post_mean, post_var)
                self.q_dists[s][a] = beta(pre_alpha, pre_beta)

            s = next_state
            t += 1
        return


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

    for ep in range(20):
        agent.train_one_episiode()

    agent.q_dists