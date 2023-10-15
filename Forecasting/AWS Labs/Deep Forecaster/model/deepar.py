import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from likelihood_layer import *
from loss_and_metrics import Loss


class DeepAR(nn.Module):
    def __init__(self, params):
        '''
        Recurrent network with lagged inputs and covariates to forecast future values
        '''
        super(DeepAR, self).__init__()
        self.params = params
        self.embedding_layer = nn.Linear(1, params['embedding_size'])
        self.lstm = nn.LSTM(input_size = params['input_size'] + params['embedding_size'],
                            hidden_size=params['lstm_hidden_size'],
                            num_layers=params['lstm_num_layers'],
                            bias = True,
                            batch_first=True,
                            dropout=params['lstm_dropout'])
        self.likelihood = params['likelihood']
        self.lr = params['lr']

        if params['data_type'] == 'continuous':
            self.likelihood_layer = Gaussian_layer(params['lstm_hidden_size'], 1)
        elif params['data_type'] == 'count':
            self.likelihood_layer = NegativeBinomial_layer(params['lstm_hidden_size'], 1)


    def forward(self, X, y, X_forecast):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()
            X_forecast = torch.from_numpy(X_forecast).float()

        num_ts, seq_len, _ = X.size()
        _, num_horizon, num_feats = X_forecast.size()

        y_f = None
        pred = []
        mus = []
        sigmas = []
        hidden, cell = None, None

        for t in range(seq_len + num_horizon):
            if t < seq_len:
                y_f = y[:, t].view(-1, 1) # dim (y[t], 1)
                y_embed = self.embedding_layer(y_f).view(num_ts, -1)
                x = X[:, t, :].view(num_ts, -1)
            else:
                y_embed = self.embedding_layer(y_f).view(num_ts, - 1)
                x = X_forecast[:, t - seq_len, :].view(num_ts, -1)
            x = torch.cat([x, y_embed], dim = 1) # num_ts, num_feats + embedding_dim
            lstm_input = x.unsqueeze(1) # (1, num_ts, num_feats + embedding_dim)

            if hidden is None and cell is None:
                output, (hidden, cell) = self.lstm(lstm_input)
            else:
                output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))


            hiddens = hidden[-1, :, :]
            hiddens = F.relu(hiddens)
            mu, sigma = self.likelihood_layer(hiddens)

            mus.append(mu.view(-1, 1))
            sigmas.append(sigma.view(-1, 1))

            if self.likelihood == 'continuous':
                y_f = Gaussian_sampling(mu, sigma)
            elif self.likelihood == 'count':
                alpha_curr = sigma
                mu_curr = mu
                y_f = NegativeBinomial_sampling(mu_curr, alpha_curr)

            if t >= seq_len - 1 and t < num_horizon + seq_len - 1:
                pred.append(y_f)

        pred = torch.cat(pred, dim = 1).view(num_ts, -1)
        mu_ = torch.cat(mus, dim = 1).view(num_ts, -1)
        sigma_ = torch.cat(sigmas, dim = 1).view(num_ts, -1)
        return pred, mu_, sigma_