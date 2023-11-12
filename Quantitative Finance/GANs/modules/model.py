import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from ..transform.DAIN import DAIN_Layer


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Creates a temporal block.
    Args:
        n_inputs (int): number of inputs.
        n_outputs (int): size of fully connected layers.
        kernel_size (int): kernel size along temporal axis of convolution layers within the temporal block.
        dilation (int): dilation of convolution layers along temporal axis within the temporal block.
        padding (int): padding
        dropout (float): dropout rate
    Returns:
        tuple of output layers
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if padding == 0:
            self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.5)
        self.conv2.weight.data.normal_(0, 0.5)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.5)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out, self.relu(out + res)


class InverseDAIN_layer(nn.Module):
    def __init__(self, dain_layer):
        super(InverseDAIN_layer, self).__init__()
        self.dain_layer = dain_layer

    def forward(self, x):
        mean = self.dain_layer.mean_layer.weight.data
        std = torch.sqrt(self.dain_layer.scaling_layer.weight.data)
        return x * std + mean

class Generator(nn.Module):
    """Generator: 3 to 1 Causal temporal convolutional network with skip connections.
       This network uses 1D convolutions in order to model multiple timeseries co-dependency.
    """
    def __init__(self, mode = 'full', mean_lr = 1e-6, gate_lr = 0.001,
                 scale_lr = 0.0001, seq_len = 127, train_mode = True, DAIN = False):
        super(Generator, self).__init__()
        self.train_mode = train_mode
        self.adaptive_normalize = DAIN
        self.tcn = nn.ModuleList([TemporalBlock(3, 80, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.normalize = DAIN_Layer(mode=mode, mean_lr=mean_lr, gate_lr=gate_lr,
                                    scale_lr=scale_lr, input_dim = 3)
        self.last = nn.Conv1d(80, 1, kernel_size=1, stride=1, dilation=1)
        # self.fc = nn.Linear(1, seq_len)

    def forward(self, x):
        # Expect input to be (num_samples/batch_size, features_dim, num_features)
        if self.train_mode:
          if self.adaptive_normalize:
            x = self.normalize(x)

        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


class Discriminator(nn.Module):
    """Discrimnator: 1 to 1 Causal temporal convolutional network with skip connections.
       This network uses 1D convolutions in order to model multiple timeseries co-dependency.
    """
    def __init__(self, seq_len, mode = 'full', mean_lr = 1e-6, gate_lr = 0.001,
                 scale_lr = 0.0001,conv_dropout=0.05, DAIN = False):
        super(Discriminator, self).__init__()
        self.tcn = nn.ModuleList([TemporalBlock(1, 80, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.adaptive_normalize = DAIN
        self.normalize = DAIN_Layer(mode=mode, mean_lr=mean_lr, gate_lr=gate_lr, scale_lr=scale_lr, input_dim = 1)
        self.last = nn.Conv1d(80, 1, kernel_size=1, dilation=1)
        self.to_prob = nn.Sequential(nn.Linear(seq_len, 1), nn.Sigmoid())

    def forward(self, x):
        if self.adaptive_normalize:
          x = self.normalize(x)
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        x = self.to_prob(x)
        return x.squeeze()