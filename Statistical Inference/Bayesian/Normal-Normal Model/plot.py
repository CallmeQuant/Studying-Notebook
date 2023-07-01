import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.pyplot as plt

def plot_normal_normal(mean, sd, sigma=None, y_bar=None, n=None, prior=True, 
                       likelihood=True, posterior=True, ax = None):
    if ax is None:
        plt.figure(figsize = (10, 5))
        ax = plt.axes()
    if y_bar is None or n is None or sigma is None:
        print("To plot the posterior, specify sigma for the likelihood, data ybar and n")
    
    post_mean = ((sigma**2)*mean + (sd**2)*n*y_bar) / (n*(sd**2) + (sigma**2))
    post_var = ((sigma**2)*(sd**2)) / (n*(sd**2) + (sigma**2))
    
    x = np.linspace(min(mean - 4*sd, y_bar - 4*sigma/np.sqrt(n)), max(mean + 4*sd, y_bar + 4*sigma/np.sqrt(n)), 100)
    
    if prior:
        ax.plot(x, norm.pdf(x, mean, sd),  label='prior')
        ax.fill_between(x, norm.pdf(x, mean, sd), alpha=0.5)
    
    if y_bar is not None and n is not None and sigma is not None and likelihood:
        ax.plot(x, norm.pdf(x, y_bar, sigma/np.sqrt(n)), label='(scaled) likelihood')
        ax.fill_between(x, norm.pdf(x, y_bar, sigma/np.sqrt(n)), alpha=0.5)
    
    if y_bar is not None and n is not None and sigma is not None and posterior:
        ax.plot(x, norm.pdf(x, post_mean, np.sqrt(post_var)),  label='posterior')
        ax.fill_between(x, norm.pdf(x, post_mean, np.sqrt(post_var)), alpha=0.5)
    
    ax.legend()
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('density')

# Testing 
sample_size = [5, 10, 20, 30]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 7))
for s, ax in zip(sample_size, axes.flatten()):
  plot_normal_normal(mean = 6.5, sd = 0.4, sigma = 0.5,
                   y_bar = 5.735, n = s, ax = ax)
  ax.set_title(f"Sample size = {s}")
  plt.tight_layout()
