import numpy as np
from typing import Callable
def rejection_sampling(target_dist: Callable, proposal_dist: Callable, proposal_sampler: Callable, 
                       k: float, n_samples: int) -> np.ndarray:
    """
    Perform rejection sampling to generate samples from target distribution.

    Parameters
    ----------
    target_dist: callable, target distribution from which to generate samples
    proposal_dist: callable, proposal distribution used to generate samples
    proposal_sampler: callable, function to generate samples from proposal distribution
    k: float, scaling constant such that k * proposal_dist(x) >= target_dist(x) for all x
    n_samples: int, number of samples to generate

    Returns
    -------
    numpy array of shape (n_samples,), generated samples
    """
    samples = []
    while len(samples) < n_samples:
        x = proposal_sampler()
        u = np.random.uniform(0, k * proposal_dist(x))
        if u <= target_dist(x):
            samples.append(x)
    return np.array(samples)
