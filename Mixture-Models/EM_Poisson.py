import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import random
from ultis_pois import training_EM


# Suppose the unknown true set of parameters that we don't know is (12, 2, 0.25, 0.75), we generate the simulated data with these parameters
# and provide initial guess on these parameters

N = 1000 # Number of observations
mu0 = 12
mu1 = 2
pi0 = 0.25
pi1 = 1 - pi0

# Generate simulated poisson data
z0 = poisson.rvs(mu0, size = round(N * pi0))
z1 = poisson.rvs(mu1, size = round(N * pi1))

X = np.concatenate((z0, z1))

# Plotting
x = np.arange(0, 100, 0.5)
fig, ax = plt.subplots(1, 1)
ax.plot(x, poisson.pmf(x, mu0), color = 'r', lw = 1)
ax.plot(x, poisson.pmf(x, mu1), color = 'b', lw = 1)
plt.show()

# # Testing functionality of all functions in our general training EM algorithm
mu = list(random.sample(list(np.arange(1, 101)), 2))
pi0 = np.random.uniform(0, 1, 1).tolist()
pi = [pi0[0], 1- pi0[0]]
# num_data_points = len(X)
# posteriors = compute_post(X, mu, pi)
# probs = update_pi(posteriors, num_data_points)
# mus = update_mu(X, posteriors)
# lowerbound_new = compute_lower_bound(X, mus, probs)
# print(len(posteriors))
# print(lowerbound_new)
# print(probs)
# print(mus)

best_mu, best_prob = training_EM(X, mu, pi)
print("The optimal value for rate parameters of Poisson distribution: ", best_mu)
print("The optimal value for rate parameters of Poisson distribution: ", best_prob)



