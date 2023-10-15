import torch
from torch import nn

def Gaussian_sampling(mu, sigma):
    pdf = torch.distributions.normal.Normal(mu, sigma)
    return pdf.sample(mu.size())

def NegativeBinomial_sampling(mu, alpha):
    # Since torch does not use parameterization in the paper, we will use the conditional expection for this sample
    var = mu + mu * mu * alpha
    return mu + torch.randn(mu.size()) * torch.sqrt(var) # inject some dispersion



class Gaussian_layer(nn.Module):

    def __init__(self, hidden_size, output_size, init_weight = 'xavier'):
        '''
        Gaussian Likelihood Supports Continuous Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(Gaussian_layer, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)
        # self.init_weight = init_weight

        # if init_weight == 'xavier':
        #     # initialize weights
        #     nn.init.xavier_uniform_(self.mu_layer.weight)
        #     nn.init.xavier_uniform_(self.sigma_layer.weight)
        # if init_weight == 'kaiming':
        #     nn.init.kaiming_uniform_(self.mu_layer.weight)
        #     nn.init.kaiming_uniform_(self.sigma_layer.weight)

    def forward(self, hidden):
        _, hidden_size = hidden.size()
        sigma_t = torch.log(1 + torch.exp(self.sigma_layer(hidden))) + 1e-6
        sigma_t = sigma_t.squeeze(0)
        mu_t = self.mu_layer(hidden).squeeze(0)
        return mu_t, sigma_t

class NegativeBinomial_layer(nn.Module):

    def __init__(self, hidden_size, output_size, init_weight = 'xavier'):
        '''
        Gaussian Likelihood Supports Continuous Data
        Args:
        hidden_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(NegativeBinomial_layer, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)
        # self.init_weight = init_weight

        # if init_weight == 'xavier':
        #     # initialize weights
        #     nn.init.xavier_uniform_(self.mu_layer.weight)
        #     nn.init.xavier_uniform_(self.sigma_layer.weight)
        # if init_weight == 'kaiming':
        #     nn.init.kaiming_uniform_(self.mu_layer.weight)
        #     nn.init.kaiming_uniform_(self.sigma_layer.weight)

    def forward(self, hidden):
        _, hidden_size = hidden.size()
        sigma_t = torch.log(1 + torch.exp(self.sigma_layer(hidden))) + 1e-6
        # sigma_t = sigma_t.squeeze(0)
        # mu_t = self.mu_layer(hidden).squeeze(0)
        mu_t = torch.log(1 + torch.exp(self.mu_layer(hidden)))
        return mu_t, sigma_t