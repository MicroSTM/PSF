from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *


class DNN(torch.nn.Module):
    """
    DNN baseline for 
        i) motion prediction or 
        ii) goal recognition
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 latent_dim=128,
                 network_type='LSTM',
                 activation='linear'):
        super(DNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.network_type = network_type
        self.activation = activation

        self.fc1 = nn.Linear(input_dim, latent_dim)
        if network_type == 'LSTM':
            self.lstm = nn.LSTMCell(latent_dim, latent_dim)
        else:
            self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.pred_linear = nn.Linear(latent_dim, output_dim)
        # init weights
        self.apply(weights_init)
        if network_type == 'LSTM':
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)

    
    def forward(self, 
                state, 
                hidden=None):
        x = F.relu(self.fc1(state))
        if self.network_type == 'LSTM':
            hidden = self.lstm(x, hidden)
            x = hidden[0]
        else:
            x = F.relu(self.fc2(x))
        x = self.pred_linear(x)
        if self.activation == 'linear':
            pred = x
        elif self.activation == 'log_softmax':
            pred = F.log_softmax(x, dim=-1)
        elif self.activation == 'sigmoid':
            pred = F.sigmoid(x)
        else:
            raise ValueError('No such activation!')
        return pred, hidden

