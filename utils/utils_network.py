from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable


def weights_init(m):
    """
    initializaing weights
    """
    initrange = 0.1
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)


def update_network(loss, optimizer):
    """update network parameters"""
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def save_model(model, path):
    """save trained model parameters"""
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """load trained model parameters"""
    model.load_state_dict(dict(torch.load(path)))


def to_variable(tensor, device):
    return tensor.to(device)


def reset_hidden_state(dim, device):
    if dim is None:
        return None
    hx = torch.zeros(dim).to(device)
    cx = torch.zeros(dim).to(device)
    return (hx, cx)


def detach_hidden_state(hidden):
    return (hidden[0].detach(), hidden[1].detach()) if hidden is not None \
        else None
