import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from likelihood_layer import *
from loss_and_metrics import Loss
class NoiseRNN(nn.Module):
    """
    Generates noise from RNN architecture that takes input features.
    """
    def __init__(self, params):
        super(NoiseRNN, self).__init__()
        self.rnn_cell = nn.LSTM(params['input_size'], params['noise_hidden_size'],
                                params['noise_num_layers'], bias = True, batch_first=True)
        self.affine_transform = nn.Linear(params['noise_hidden_size'], 1)

    def forward(self, X):
        num_ts, num_feats = X.shape
        X = X.unsqueeze(1)
        _, (hidden, cell) = self.rnn_cell(X)
        hiddens = hidden[-1, :, :] # dim (num_ts, random effects)
        hiddens = F.relu(hiddens)
        sigma_t = self.affine_transform(hiddens)
        sigma_t = torch.log(1 + torch.exp(sigma_t)) # Sofplus to make sure sigma_t is positive
        return sigma_t.view(-1, 1)

class GlobalFactor(nn.Module):
    """
    Generates global effects (fixed effects or common patterns) modelled by RNNs
    given the input features. Later this will play as pivotal
    components in the linear combination of deep factors.
    """
    def __init__(self, params):
        super(GlobalFactor, self).__init__()
        self.rnn_cell = nn.LSTM(params['input_size'], params['global_hidden_size'],
                                params['global_num_layers'],
                                bias = True, batch_first=True)
        self.latent_deep_factor = nn.Linear(params['global_hidden_size'], params['global_num_factors'])

    def forward(self, X):
        num_ts, num_feats = X.shape
        X = X.unsqueeze(1)
        _, (hidden, cell) = self.rnn_cell(X)
        hiddens = hidden[-1, :, :]  # dim (num_ts, global effects)
        hiddens = F.relu(hiddens)
        gt = hiddens
        return gt.view(num_ts, -1)

class DeepFactorRNN(nn.Module):
    def __init__(self, params):
        super(DeepFactorRNN, self).__init__()
        self.generate_noise = NoiseRNN(params)
        self.global_factor = GlobalFactor(params)
        self.affine = nn.Linear(params['global_hidden_size'], params['global_num_factors'])

    def forward(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        num_ts, num_periods, num_feats = X.size()
        mus = []
        sigmas = []
        for t in range(num_periods):
            global_factor_t = self.global_factor(X[:, t, :])
            fixed_effect_t  = self.affine(global_factor_t)
            fixed_effect_t = fixed_effect_t.sum(dim = 1).view(-1, 1)
            sigma_t = self.generate_noise(X[:, t, :])
            mus.append(fixed_effect_t)
            sigmas.append(sigma_t)
        mus = torch.cat(mus, dim = 1).view(num_ts, num_periods)
        sigmas = torch.cat(sigmas, dim = 1).view(num_ts, num_periods) + 1e-6 # prevent nonnegative
        return mus, sigmas

    def sample(self, X, num_samps = 100):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        mu, sigma = self.forward(X)
        num_ts, num_periods = mu.size()
        z = torch.zeros(num_ts, num_periods)
        for _ in range(num_samps):
            likelihood = torch.distributions.normal.Normal(loc = mu, scale = sigma)
            z_samp = likelihood.sample().view(num_ts, num_periods)
            z += z_samp

        z = z / num_samps
        return z