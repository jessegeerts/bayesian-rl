import numpy as np
import matplotlib.pyplot as plt

c = 1  # prior variance
s = 1  # noise variance
q = .01  # diffusion variance
K = 6  # number of microstimuli
sigma = .08  # width of microstimuli
decay = .985  # Trace decay
lr = .3  # learning rate for standard td
TD = False  # use standard TD
gamma = .98

# make stimulus

n_trials = 10
n_features = 10
trial_length = 10

X = np.zeros((n_trials * trial_length, n_features))
for j in range(n_trials):
    for i in range(trial_length):
        X[j*trial_length + i, i] = 1

# Reward
r_trial = np.zeros(trial_length)
r_trial[-1] = 1
r = np.tile(r_trial, reps=n_trials)

# initialisation
C = np.eye(n_features) * c  # prior covariance matrix

N, D = np.shape(X)
w = np.zeros(D)

model = {}

for n in range(N-1):

    model[n] = {'w': w}
    model[n]['V'] = np.dot(X[n, :], w)

    Q = q * np.eye(D)
    h = X[n, :] - gamma * X[n+1, :]
    rhat = np.dot(h, w)
    dt = r[n] - rhat
    C = C + Q  # a priori covariance
    P = np.dot(h, np.matmul(C, h.T)) + s  # residual covariance

    K = np.matmul(C, h) / P
    w0 = w

    w = w + K * dt  # weight update

    C = C - np.matmul(np.outer(K, h), C)  # posterior covariance

    model[n]['C'] = C
    model[n]['K'] = K
    model[n]['PE'] = dt
    model[n]['rhat'] = rhat


# TODO: why negative? why this weird turnaround? maybe ask sam?