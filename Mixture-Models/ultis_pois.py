import numpy as np
from typing import List, Tuple, Dict, Union
import numpy.typing as npt


def poisson_pmf(x: float, m: Union[int, float]) -> float:
    """
    A function to return p.m.f of poisson distribution

    Parameters
    ----------
    x: float
        each observation in the data
    m: Union[int, float]
        rate parameter of Poisson distribution

    Returns
    -------
    pmf: float
        Probability mass function of Poisson distribution
    """
    pmf = (m ** x) * np.exp(-m) / np.math.factorial(x)
    return pmf


def compute_post(X: npt.NDArray, mus: List[float], probs: List[float]) -> List[List[float]]:
    """
    Compute the posterior of p(Z_{i} = k \mid X_{i}, \theta_{t-1}). This quantity is
    also the q(Z) that make the lower bound tight

     Parameters
    ----------
    X:  np.ndarray
        A two-dimensional array contains data from two Poisson distributions that are stacked horizontally.
    mus: List[float]
        Parameter of two Poisson distributions
    probs: List[float]
        Probability of each class (mixing probabilities)

    Returns
    -------
    posterior: List[List[float]
        Posterior of each class given the observations
    """
    post = []
    for data in X:
        temp = []
        # Compute P(X_{i} \mid Z_{i} = k, \theta_{t-1}) * \pi_{k}
        for i in range(len(mus)):
            likelihood = poisson_pmf(data, mus[i]) * probs[i]
            temp.append(likelihood)
        # Compute the posterior
        temp = [x / sum(temp) for x in temp]
        post.append(temp)
    return np.array(post).T.tolist()


def update_pi(posterior_distribution: List[float], num_data_points: int) -> Tuple[float]:
    """
    Updating the probability of class \pi_{k}

    Parameters
    ----------
    posterior_distribution: List[float]
        Posterior distributions of each class given the data and parameters,
        length of list should be two where length of all posterior within each class should equal number of data points
    num_data_points: int
        Number of observations in our data

    Returns
    -------
    new parameters: List[float]
        Probability of each class w.r.t observations
    """
    # \sum_{i=1}^{N}  p(Z_{i} = k \mid X_{i}, \theta_{t-1}) / N
    pi0 = sum(posterior_distribution[0]) / num_data_points
    return pi0, 1 - pi0


def update_mu(X: npt.NDArray, posterior_distribution: List[List[float]]) -> List[float]:
    """
    Updating the parameter \mu_k of each poisson distribution in mixture

    Parameters
    ----------
    X:  np.ndarray
        A two-dimensional array contains data from two Poisson distributions that are stacked horizontally.
    posterior_distribution: List[float]
        Posterior distributions of each class given the data and parameters,
        length of list should be two where length of all posterior within each class should equal number of data points

    Returns
    -------
    new parameters: List[float]
        Parameters of each Poisson distribution
    """
    mu_update = []
    for i in range(len(posterior_distribution)):
        numerator = [x * y for x, y in zip(X, posterior_distribution[i])]

        # compute mu
        mu_ = sum(numerator) / sum(posterior_distribution[i])
        mu_update.append(mu_)

    return mu_update


def compute_lower_bound(X: npt.NDArray, mus: List[float], probs: List[float]):
    """
    Compute the variational lower bound

    Parameters
    ----------
    X:  np.ndarray
        A two-dimensional array contains data from two Poisson distributions that are stacked horizontally.
    mus: List[float]
        Parameter of two Poisson distributions
    probs: List[float]
        Probability of each class (mixing probabilities)

    Returns
    -------
    Lower bound: float
        The value of lower bound after updating the parameters
    """
    loglik = []
    for data in X:
        likelihood = []
        for i in range(len(mus)):
            lik = poisson_pmf(data, mus[i]) * probs[i]
            likelihood.append(lik)

        loglik_ = np.log(sum(likelihood))
        loglik.append(loglik_)

    lowerbound = sum(loglik)
    return lowerbound


def training_EM(X: npt.NDArray, mus: List[float], probs: List[float], num_iter: int=50, tol: float=1e-5) -> Tuple[float]:
    """EM Algorithm
    A function to train EM algorithm. The algorithm procedure is as follows:
    1. With initial parameters, we compute the posterior of the hidden variable (in our example, this is the class)
    2. Using the computed posterior, updating the parameters (the mixing probability and the mean of poisson distribution)
    3. With newly updated parameters, recompute the lowerbound and check for its convergence. Note that
    if we have run through all iterations and still not obtain the optimal values for parameters, we can increase the number of iterations or re-initialize the guess of parameters.

    References: Pattern Recognition and Machine Learning (Bishop, 2006)

    Parameters
    ----------
    X:  np.ndarray
        A two-dimensional array contains data from two Poisson distributions that are stacked horizontally.
    mus: List[float]
        Parameter of two Poisson distributions
    probs: List[float]
        Probability of each class (mixing probabilities)
    num_iter: int
        Number of iterations
    tol: float
        Tolerance for checking convergence

    Returns
    -------
    estimations: Tuple[float]
        The optimal estimations for parameters in mixture Poisson models
    """
    delta = 1
    lowerbound = []
    counter = 0
    num_data_points = len(X)

    while counter < num_iter:

        # E-step
        # Compute the posterior of latent variables given current parameters and data
        posteriors = compute_post(X, mus, probs)

        # M-step
        # Update parameters
        probs = update_pi(posteriors, num_data_points)
        mus = update_mu(X, posteriors)

        # Construct new lower bound to determine convergence
        lowerbound_new = compute_lower_bound(X, mus, probs)
        lowerbound.append(lowerbound_new)

        # Compare the loglikelihood
        if len(lowerbound) > 1:
            lowerbound_old = lowerbound[len(lowerbound) - 2]
            lowerbound_curr = lowerbound[len(lowerbound) - 1]
            delta = np.abs(lowerbound_curr - lowerbound_old)

            if delta < tol:
                print(f'EM algorithm has converged after {counter} iterations ')
                return mus, probs
                break

        counter += 1
    print(
        f"EM algorithm can't converge to optimal parameters in {num_iter} iterations. Consider to increase number of iterations!")
    return mus, probs