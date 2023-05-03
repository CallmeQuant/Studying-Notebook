import os
import torch
import numpy as np
from sklearn.base import BaseEstimator


def gaussian_likelihood(X, mu, sigma):
    """
    Computes the Gaussian pdf

    Parameters
    ----------
    X: Features of data
      Torch tensor
    mu: Mean parameter of Gaussian likelihood
      Float
    sigma: Standard deviation of Gaussian likelihood
      Float

    Return
    ------
    Gaussian pdf
    """
    return (2 * np.pi * sigma ** 2) ** (-0.5) * torch.exp((- (X - mu) ** 2) / (2 * sigma ** 2))
class Naive_Bayes(BaseEstimator):
    def __init__(self, offset=1):
        """Init function for the naive bayes class"""
        self.offset = offset

    def fit(self, X, y, **kwargs):
        """
        Fit Naive Bayes classifier given features tensor and labels tensor

        Parameters
        ----------
        X: Features of data
            Torch tensor
        y: Labels of data
            Torch tensor

        Returns
        -------
        Likelihoods: Probabilities of each observation given each class/label
            List[Torch tensor]
        """
        self.is_categorical = kwargs['is_categorical']
        self.num_feats = X.size(1)
        # size = X.size(0)
        y_vals_unique = y.unique()
        self.num_class = len(y_vals_unique)
        # Probability of each class in the training set
        self.class_probs = y.int().bincount().float() / len(y)

        feats_vals_max = torch.zeros((self.num_feats,), dtype=torch.int32)
        for i in range(self.num_feats):
            feats_vals_max[i] = X[:, i].max()

        # Initialize list to store p(x_j | c_i)
        likelihoods = []
        for i in range(self.num_class):
            likelihoods.append([])
            # Index to group samples by class
            idx = torch.where(y == y_vals_unique[i])[0]  # torch.where returns a tuple
            curr_class = X[idx]
            class_size = curr_class.size(0)
            for j in range(self.num_feats):
                # Store all classes
                likelihoods[i].append([])
                if self.is_categorical[j]:
                    for k in range(feats_vals_max[j] + 1):
                        # Count number of observations of each feature given the class
                        prob_feat_in_class = (torch.where(curr_class[:, j])[0].size(0) + self.offset) / class_size
                        likelihoods[i][j].append(prob_feat_in_class)

                else:
                    feats_class = curr_class[:, j]
                    mean = feats_class.mean()
                    sigma = feats_class.std()  # set 'correction = 0' for not using Bessel's correction
                    likelihoods[i][j] = [mean, sigma]

            self.likelihoods = likelihoods
            return self.likelihoods
    def predict(self, X):
        """

        Parameters
        X: Features of data
            Torch tensor

        Returns:
        -------
        Predicted labels for each obseravtion
            Torch tensor

        """
        if len(X.size()) == 1:
            X = X.unsqueeze(0)  # dim (1, D)

        num_obs = X.size(0)
        pred = torch.zeros((num_obs, self.num_class), dtype=torch.float32)
        for k in range(num_obs):
            curr_obs = X[k]
            for i in range(self.num_class):
                pred[k][i] = self.class_probs[i]  # Set prior probability for ith class p(c_i)
                prob_feat_in_class = self.likelihoods[i]  # likelihoods for ith class p(x_j | c_i)
                for j in range(self.num_feats):
                    if self.is_categorical[j]:
                        pred[k][i] *= prob_feat_in_class[j][curr_obs[j].int()]
                    else:
                        mean, sigma = prob_feat_in_class[j]
                        pred[k][i] *= gaussian_likelihood(curr_obs[j], mean, sigma)

        return pred.argmax(dim=1)  # Return maximum probability among all classes