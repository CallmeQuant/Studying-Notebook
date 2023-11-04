import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import partial
import numpy as np

# tqdm for loading bars
from tqdm.notebook import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('Device: ', device)

class Embedding(nn.Module):
    """
    Encodes the static and dynamic features (states) using 1D covolutional layers
    Note that the operation of Conv1D is the same as nn.Linear if we think set out_features
    equal to the out_channels (i.e., embedding_size).
    """
    def __init__(self, input_size, embedding_size):
        super(Embedding, self).__init__()
        self.conv = nn.Conv1d(input_size, embedding_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output # shape (batch_size, embedding_size/hidden_size, seq_len)

class Attention_layer(nn.Module):
    """
    Compute the additive attention over the input nodes given the current states (hidden states)
    """

    def __init__(self, hidden_size):
        super(Attention_layer, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size),
                                          device=device, requires_grad=True))

    def forward(self, static_feats_hidden, dynamic_feats_hidden, decoder_hidden):
        batch_size, hidden_size, _ = static_feats_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_feats_hidden)
        hidden = torch.cat((static_feats_hidden, dynamic_feats_hidden, hidden), 1)

        # Broadcast first dimensions to match batch_size, so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attentions = torch.bmm(v, torch.tanh(torch.bmm(W, hidden))) # (W @ hidden) -> (batch_size, hidden_size, seq_len) -> v -> (batch_size, 1, seq_len)
        attentions = F.softmax(attentions, dim=2)  # (batch, 1, seq_len)
        if attentions.ndimension() < 3: # Make sure the shape to be (batch_size, 1, seq_len)
            attentions = attentions.unsqueeze(1)
        return attentions

class Pointer(nn.Module):
    """
    Compute the next state given the previous state and input embeddings.
    """

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
                                          device=device, requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.encoder_attn = Attention_layer(hidden_size)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):
        """
        Parameters
        ----------
        static_hidden: torch tensor of shape (batch_size, hidden_size, seq_len)
            Embedded static features
        dynamic_hidden: torch tensor of shape (batch_size, hidden_size, seq_len)
            Embedded dynamic features
        decoder_hidden: torch tensor of shape (batch_size, hidden_size, seq_len)
            current state of decoder
        last_hh: tensor of shape (num_layers, batch_size, hidden_size)
            last hidden state of RNN (here is GRU)
        """

        # input for RNN in torch must have shape (batch_size, seq_len, hidden_size) for batched input
        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh) # rnn_output: (batch_size, 1, hidden_size)
        rnn_out = rnn_out.squeeze(1) # (batch_size, hidden_size)
        # # output of decoder shape
        # print(rnn_out.shape)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh)

        # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out) # (batch_size, seq_len)

        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # (batch_size, 1, hidden_size/num_feats)

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)  # (batch_size, 2*num_feats, seq_len)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        # Get (batch_size, 1, seq_len) then squeezed
        # Unnormalized probabilities of pointing to each element in input
        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)

        return probs, last_hh # probs shape (batch_size, seq_len), last_hh shape (num_layers, batch_size, hidden_size)

