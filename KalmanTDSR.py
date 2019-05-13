# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from agent import KalmanSR
from environment import SimpleMDP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from dynamic_programming import value_iteration
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
# %matplotlib notebook

env = SimpleMDP(4)

env.create_graph()

# +
plt.figure()

#positions = {0:(0,0), 1:(1,0), 2:(2,0), 3:(3,0), 4:(4,0)}

env.show_graph(layout=positions)
# -

# # Learning the SR using a kalman filter 

# +
transition_noise = .005 * np.eye(env.nr_states**2)
gamma=.9
prior_M = np.eye(env.nr_states).flatten()
prior_covariance = np.eye(env.nr_states**2) #np.ones((env.nr_states**2, env.nr_states**2))
observation_noise_variance = np.eye(env.nr_states)

M = prior_M
covariance = prior_covariance

def get_feature_representation(state_idx):
    """Get one-hot feature representation from state index.
    """
    if env.is_terminal(state_idx):
        return np.zeros(env.nr_states)
    else:
        return np.eye(env.nr_states)[state_idx]



# -



for episode in tqdm(range(10)):
    env.reset()
    t = 0
    s = env.get_current_state()
    features = get_feature_representation(s)

    while not env.is_terminal(env.get_current_state()) and t < 1000:
        a = 1

        next_state, reward = env.act(a)
        next_features = get_feature_representation(next_state)
        H = features - gamma * next_features  # Temporal difference features

        #Prediction step;
        a_priori_covariance = covariance + transition_noise

        # Compute statistics of interest;
        feature_block_matrix = np.kron( H, np.eye(env.nr_states))  # probably the fault is here
        phi_hat = np.matmul(feature_block_matrix, M)
        delta_t = features - phi_hat
        parameter_error_cov = np.matmul(a_priori_covariance, feature_block_matrix.T)  # maybe has to be transpose
        residual_cov = np.matmul(np.matmul(feature_block_matrix, a_priori_covariance), feature_block_matrix.T) + observation_noise_variance

        # Correction step;
        kalman_gain = np.matmul(parameter_error_cov, np.linalg.inv(residual_cov))
        delta_M = np.matmul(kalman_gain, delta_t)
        M += delta_M

        covariance = a_priori_covariance - np.matmul(np.matmul(kalman_gain, residual_cov), kalman_gain.T)

        s = next_state
        features = get_feature_representation(s)

        t += 1
np.around(M.reshape(env.nr_states,-1), decimals=3)

M

np.diag(covariance)

# how does the 1--> 2 predictiveness covary with the 2-->3 predictiveness? 
M[1]

np.nonzero(covariance[0])

covariance[1,7]

plt.figure()
plt.imshow(covariance[:-env.nr_states, :-env.nr_states])
plt.colorbar()


covariance[0]

M

covariance[0]








