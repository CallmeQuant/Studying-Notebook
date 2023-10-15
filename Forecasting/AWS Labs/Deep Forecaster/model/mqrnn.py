import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from likelihood_layer import *

class Decoder(nn.Module):
  def __init__(self, input_size,
        output_horizon,
        encoder_hidden_size,
        decoder_hidden_size,
        output_size):
    super(Decoder, self).__init__()
    self.global_mlp = nn.Linear(output_horizon * (encoder_hidden_size + input_size), \
              (output_horizon+1) * decoder_hidden_size)
    self.local_mlp = nn.Linear(decoder_hidden_size + input_size + 1, output_size)
    self.decoder_hidden_size = decoder_hidden_size

  def forward(self, hidden_t, X_forecast):
    """
    hidden_t: (1, hidden_size)
    X_forecast: (1, output_horizon, num_features)
    """
    num_ts, output_horizon, num_feats = X_forecast.size()
    num_ts, hidden_size = hidden_t.size() # (1, hidden_size)

    hidden_t = hidden_t.unsqueeze(1) # (1, 1, hidden_size)
    hidden_t = hidden_t.expand(num_ts, output_horizon, hidden_size)

    # Concatentate future covariates and last hidden states from encoder
    concat_input = torch.cat([X_forecast, hidden_t], dim = 2).view(num_ts, -1)

    contextual_input = self.global_mlp(concat_input)
    contextual_input = contextual_input.view(num_ts, output_horizon + 1, self.decoder_hidden_size)

    contextual_input_t = contextual_input[:, -1, :].view(num_ts, -1)
    contextual_input_final = contextual_input_t[:, :-1]
    contextual_input_final = F.relu(contextual_input_final)

    y = []
    for i in range(output_horizon):
        contextual_input_ = contextual_input_final[:, i].view(num_ts, -1)
        # print(f'contextual_input_ shape: {contextual_input_.size()}')
        X_forecast_ = X_forecast[:, i, :].view(num_ts, -1)
        # print(f'X_forecast_ shape: {X_forecast_.size()}')
        concat_input = torch.cat([X_forecast_, contextual_input_, contextual_input_t], dim = 1)
        # print(f'concat_input shape: {concat_input.size()}')
        out = self.local_mlp(concat_input) # (num_ts, num_quantiles)

        y.append(out.unsqueeze(1)) # (num_ts, 1, num_quantiles)

    y = torch.cat(y, dim = 1) # (num_ts/batch_size, output_horizon ,num_quantiles)
    return y


class MQRNN(nn.Module):
    def __init__(self, params):
        """
        Parameters
        ----------

        output_horizon (int): output horizons to output in prediction
        num_quantiles (int): number of quantiles interests, e.g. 0.25, 0.5, 0.75
        input_size (int): feature size
        embedding_size (int): embedding size
        encoder_hidden_size (int): hidden size in encoder
        encoder_n_layers (int): encoder number of layers
        decoder_hidden_size (int): hidden size in decoder
        """
        super(MQRNN, self).__init__()
        self.output_horizon = params['output_horizon']
        self.encoder_hidden_size = params['encoder_hidden_size']
        self.embed_layer = nn.Linear(1, params['embedding_size'])  # time series embedding
        self.encoder = nn.LSTM(params['input_size'] + params['embedding_size'], params['encoder_hidden_size'], \
                               params['num_layers'], bias=True, batch_first=True)
        self.decoder = Decoder(params['input_size'], params['output_horizon'], params['encoder_hidden_size'], \
                               params['decoder_hidden_size'], params['num_quantiles'])

    def forward(self, X, y, X_forecast):
        """
        Parameters
        ----------
        X: (num_ts, num_periods, num_feats)
        y: (num_ts, num_periods)
        X_forecast: (num_ts, seq_len, num_feats)
        """

        if isinstance(X, type(np.empty(2))):
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()
            X_forecast = torch.from_numpy(X_forecast).float()

        num_ts, num_periods, num_feats = X.size()
        y = y.unsqueeze(2) # (num_ts, num_periods, 1)
        y = self.embed_layer(y)
        x = torch.cat([X, y], dim = 2)

        _, (h, c) = self.encoder(x)
        hidden_t = h[-1, :, :]

        hidden_t = F.relu(hidden_t)
        y_pred = self.decoder(hidden_t, X_forecast)
        return y_pred





