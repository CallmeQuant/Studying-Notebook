import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import copy
import math

class Attention_layer(nn.Module):
  """Additive attention based on Luong method"""
  def __init__(self, hidden_size, input_size, seq_len, method):
    super(Attention_layer, self).__init__()
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.seq_len = seq_len
    self.method = method
    self.embedding_hidden = nn.Linear(self.hidden_size, self.input_size)

    if self.method not in ['general', 'concat']:
      raise ValueError(self.method, 'is not an appropriate attention method')
    if self.method == 'general':
      self.attention = torch.nn.Linear(self.seq_len, seq_len)
    elif self.method == 'concat':
      self.attention = torch.nn.Linear(self.hidden_size + self.input_size, hidden_size)
      self.v = nn.Parameter(torch.rand(hidden_size))
      self.v = nn.Parameter(torch.rand(hidden_size))
      stdv = 1. / np.sqrt(self.v.size(0))
      self.v.data.normal_(mean=0, std=stdv)

  def forward(self, hidden, inputs):
    attn_energy = self.score(hidden, inputs)
    return F.softmax(attn_energy, dim = 1).unsqueeze(1) # [B*1*T]

  def score(self, hidden, inputs):
    if self.method == 'concat':
      if hidden.ndimension() < 3:
        hidden = hidden.unsqueeze(1).expand(-1, X.size(1), -1)
      energy = torch.tanh(self.attention(torch.cat([hidden, inputs], 2))) # [B*T*H]
      energy = torch.transpose(energy, 1, 2) # [B*H*T]
      v = self.v.repeat(inputs.size(0), 1).unsqueeze(1)  # [B*1*H]
      energy = torch.bmm(v, energy)  # [B*1*T]
      energy = energy.squeeze(1)
    elif self.method == 'general':
      if hidden.size(1) != inputs.size(2):
        embedded_hidden = self.embedding_hidden(hidden)
      energy = self.attention(torch.bmm(inputs, embedded_hidden.unsqueeze(2)).squeeze(2))

    return energy # [B*T]

class GRU(nn.Module):
    def __init__(self, num_layers, hidden_dim, seq_len, num_stocks, drop_out = 0.2,
                bidirectional = False, use_attention = True, lb = 0, ub = 0.2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lb = lb
        self.ub = ub
        # Attention layer
        self.use_attention = use_attention
        self.attention = Attention_layer(hidden_dim, num_stocks, seq_len, 'concat') if use_attention else None
        # GRU layer
        self.gru = nn.GRU(
            num_stocks, self.hidden_dim, num_layers = self.num_layers,
            batch_first=True, bidirectional = bidirectional
        )
        self.dropout = nn.Dropout(drop_out)
        self.scale = 2 if bidirectional else 1
        self.fc = nn.Linear(self.hidden_dim * self.scale*2, num_stocks) if self.use_attention else nn.Linear(self.hidden_dim * self.scale, num_stocks)
        self.swish = nn.SiLU()

    def forward(self, x):
        # initialize hidden states bookkeeping
        h0 = torch.zeros((self.num_layers * self.scale, x.size(0), self.hidden_dim)).to(x.device)
        out, _ = self.gru(x, h0)
        h_t = out[:, -1, :]
        # Calculate the attention weights
        if self.use_attention:
          attn_weights = self.attention(h_t, x)
          context = attn_weights.bmm(out).squeeze(1) # batch_size x hidden_dim
          concat_hidden = torch.concat([h_t, context], dim = 1)
          logit = self.fc(self.dropout(concat_hidden))
        else:
          logit = self.fc(self.dropout(h_t))
        logit = F.softmax(logit, dim = -1)
        logit = torch.stack([self.rebalance(batch, self.lb, self.ub) for batch in logit])

        return logit

    def rebalance(self, weight, lb, ub):
        old = weight
        weight_clamped = torch.clamp(old, lb, ub)
        while True:
            leftover = (old - weight_clamped).sum().item()
            nominees = weight_clamped[torch.where(weight_clamped != ub)[0]]
            gift = leftover * (nominees / nominees.sum())
            weight_clamped[torch.where(weight_clamped != ub)[0]] += gift
            old = weight_clamped
            if len(torch.where(weight_clamped > ub)[0]) == 0:
                break
            else:
                weight_clamped = torch.clamp(old, lb, ub)
        return weight_clamped

class TCN(nn.Module):
    def __init__(self, n_feature, n_output, num_channels,
                 kernel_size, n_dropout, n_timestep, lb, ub):
        super(TCN, self).__init__()
        self.input_size = n_feature
        self.tcn = TemporalConvNet(n_feature, num_channels,
                                   kernel_size, dropout=n_dropout)
        self.fc = nn.Linear(num_channels[-1], n_output)
        self.tempmaxpool = nn.MaxPool1d(n_timestep)
        self.lb = lb
        self.ub = ub

    def forward(self, x):
        output = self.tcn(x.transpose(1, 2))
        output = self.tempmaxpool(output).squeeze(-1)
        out = self.fc(output)
        out = F.softmax(out, dim=1)
        out = torch.stack([self.rebalance(batch, self.lb, self.ub) for batch in out])
        return out

    def rebalance(self, weight, lb, ub):
        old = weight
        weight_clamped = torch.clamp(old, lb, ub)
        while True:
            leftover = (old - weight_clamped).sum().item()
            nominees = weight_clamped[torch.where(weight_clamped != ub)[0]]
            gift = leftover * (nominees / nominees.sum())
            weight_clamped[torch.where(weight_clamped != ub)[0]] += gift
            old = weight_clamped
            if len(torch.where(weight_clamped > ub)[0]) == 0:
                break
            else:
                weight_clamped = torch.clamp(old, lb, ub)
        return weight_clamped

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        """
        chomp_size: zero padding size
        """
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i

            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def creatMask(batch, sequence_length):
    mask = torch.zeros(batch, sequence_length, sequence_length)
    for i in range(sequence_length):
        mask[:, i, :i + 1] = 1
    return mask


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None, returnWeights=False):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    # print("Scores in attention itself",torch.sum(scores))
    if (returnWeights):
        return output, scores

    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, returnWeights=False):

        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next

        if (returnWeights):
            scores, weights = attention(q, k, v, self.d_k, mask, self.dropout, returnWeights=returnWeights)
            # print("scores",scores.shape,"weights",weights.shape)
        else:
            scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)
        # print("Attention output", output.shape,torch.min(output))
        if (returnWeights):
            return output, weights
        else:
            return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=400, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, returnWeights=False):
        x2 = self.norm_1(x)
        # print(x2[0,0,0])
        # print("attention input.shape",x2.shape)
        if (returnWeights):
            attenOutput, attenWeights = self.attn(x2, x2, x2, mask, returnWeights=returnWeights)
        else:
            attenOutput = self.attn(x2, x2, x2, mask)
        # print("attenOutput",attenOutput.shape)
        x = x + self.dropout_1(attenOutput)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        if (returnWeights):
            return x, attenWeights
        else:
            return x


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=100, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)

        pe = Variable(self.pe[:, :seq_len], requires_grad=False)

        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, input_size, seq_len, N, heads, dropout):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(input_size, seq_len, dropout=dropout)
        self.layers = get_clones(EncoderLayer(input_size, heads, dropout), N)
        self.norm = Norm(input_size)

    def forward(self, x, mask=None, returnWeights=False):
        x = self.pe(x)

        for i in range(self.N):
            if (i == 0 and returnWeights):
                x, weights = self.layers[i](x, mask=mask, returnWeights=returnWeights)
            else:
                # print(i)
                x = self.layers[i](x, mask=mask)

        if (returnWeights):
            return self.norm(x), weights
        else:
            return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, n_feature, n_timestep, n_layer, n_head, n_dropout, n_output, lb, ub):
        super().__init__()
        self.encoder = Encoder(n_feature, n_timestep, n_layer, n_head, n_dropout)
        self.out = nn.Linear(n_feature, n_output)
        self.tempmaxpool = nn.MaxPool1d(n_timestep)
        self.lb = lb
        self.ub = ub

    def forward(self, src, returnWeights=False):
        mask = creatMask(src.shape[0], src.shape[1]).to(device)
        # print(src.shape)
        if (returnWeights):
            e_outputs, weights, z = self.encoder(src, mask, returnWeights=returnWeights)
        else:
            e_outputs = self.encoder(src, mask)

        e_outputs = self.tempmaxpool(e_outputs.transpose(1, 2)).squeeze(-1)
        output = self.out(e_outputs)
        output = F.softmax(output, dim=1)
        output = torch.stack([self.rebalance(batch, self.lb, self.ub) for batch in output])
        if (returnWeights):
            return output, weights
        else:
            return output

    def rebalance(self, weight, lb, ub):
        old = weight
        weight_clamped = torch.clamp(old, lb, ub)
        while True:
            leftover = (old - weight_clamped).sum().item()
            nominees = weight_clamped[torch.where(weight_clamped != ub)[0]]
            gift = leftover * (nominees / nominees.sum())
            weight_clamped[torch.where(weight_clamped != ub)[0]] += gift
            old = weight_clamped
            if len(torch.where(weight_clamped > ub)[0]) == 0:
                break
            else:
                weight_clamped = torch.clamp(old, lb, ub)
        return weight_clamped