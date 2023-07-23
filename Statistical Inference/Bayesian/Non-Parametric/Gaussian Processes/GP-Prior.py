import numpy as np
import matplotlib.pyplot as plt

def squared_exponential_kernel(x1, x2):
  return np.exp(-1/2. * np.linalg.norm([x1 - x2], 2)**2)

def periodic_kernel(x1, x2):
  p = 3
  return np.exp(-2 * np.sin((np.pi * np.linalg.norm([x1 - x2], 1)) / p)**2)

def linear_kernel(x1, x2):
  b = 1; c = 0
  return b + ((x1 - c) * (x2 - c))

def create_cov(n, kernel_func):
  K = np.empty((n, n))
  for i in range(n):
    x1 = X[i]
    for j in range(n):
      x2 = X[j]
      K[i, j] = kernel_func(x1, x2)

  return K

fig, axes = plt.subplots(1, 3, figsize = (16, 8))
cov_funcs = [squared_exponential_kernel, periodic_kernel, linear_kernel]
n_samples = 400
n_priors  = 20
X         = np.linspace(-5, 5, n_samples)

for kernel, ax in zip(cov_funcs, axes):
  mean = np.zeros(n_samples)
  cov = create_cov(n_samples, kernel)
  for _ in range(n_priors + 1):
    f = np.random.multivariate_normal(mean, cov)
    ax.plot(np.arange(n_samples), f)

plt.show()