from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import pickle
import random
import numpy as np
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim

from models import DNN
from utils import *


X_DIM = 28
Y_DIM = 2


def prepare_training_data(trajs, max_episode_length, agent_id):
    X = []
    Y = []
    N = len(trajs)
    for episode_id in range(N):
        traj1, traj2 = trajs[episode_id][0], trajs[episode_id][1]
        T = min(len(traj1), args.max_episode_length)
        X.append([np.array(phi(traj1[t], traj2[t])) for t in range(T)])
        Y.append(1 if agent_id == 0 else 0)
        X.append([np.array(phi(traj2[t], traj1[t])) for t in range(T)])
        Y.append(0 if agent_id == 0 else 1)
    return X, Y
    

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--t-max', type=int, default=50, help='Max number of forward steps for A2C before update')
parser.add_argument('--max-episode-length', type=int, default=31, help='Maximum episode length')
parser.add_argument('--network-type', type=str, default='LSTM', help='Network type (MLP, LSTM)')
parser.add_argument('--latent-dim', type=int, default=128, help='Hidden size of LSTM cell')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch-size', type=int, default=16, help='Off-policy batch size')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
parser.add_argument('--data-dir', type=str, default='./data/training_data',help='Data directory')
# parser.add_argument('--train_ratio', type=float, default=0.75, help='Ratio of training instances')
parser.add_argument('--dataset', type=str, default='blocking', help='Dataset name')
parser.add_argument('--agent-id', type=int, default=0, help='Predicting the goal of which agent')
parser.add_argument('--train-size', type=int, default=50, help='The size of the training set')


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print (' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    checkpoint_dir = args.checkpoint_dir + '/{}_{}'.format(args.network_type, args.dataset)
    p = Path(checkpoint_dir)
    if not p.is_dir():
        p.mkdir(parents = True)

    data = pickle.load(open(args.data_dir + '/{}/data.pik'.format(args.dataset), 'rb'))
    num_episodes = min(len(data['trajs']), args.train_size)
    trajs = data['trajs'][:args.train_size]
    X, Y = prepare_training_data(trajs, args.max_episode_length, args.agent_id)
    N = len(X)
    # N_train = int(N * args.train_ratio)
    # N_test = N - N_train
    # X_train, Y_train = X[:N_train], Y[:N_train]
    # X_test, Y_test = X[N_train:], Y[N_train:]

    model = DNN(X_DIM, Y_DIM, args.latent_dim, args.network_type, activation='log_softmax')
    if args.cuda:
        model.cuda()
        NLL_loss = torch.nn.NLLLoss(reduce=False).cuda()
    else:
        NLL_loss = torch.nn.NLLLoss(reduce=False)
    optimizer = optim.RMSprop(list(model.parameters()), lr=args.lr)

    model.train()

    nb_epoch = 500
    best_loss = 1e6
    for epoch_id in range(nb_epoch):
        indices = list(range(N))
        random.shuffle(indices)
        batch_id = 0
        for start in range(0, N, args.batch_size):
            end = min(N - 1, start + args.batch_size - 1)
            batch_X = []
            batch_Y = []
            lengths = []
            for sample_id in indices[start:(end+1)]:
                batch_X.append(X[sample_id])
                batch_Y.append(Y[sample_id])
                lengths.append(len(X[sample_id]))
            T = max(lengths)
            n = len(batch_X)
            pred_loss = 0
            if args.network_type == 'LSTM':
                if args.cuda:
                    hx = Variable(torch.zeros(n, args.latent_dim).cuda())
                    cx = Variable(torch.zeros(n, args.latent_dim).cuda())
                else:
                    hx = Variable(torch.zeros(n, args.latent_dim))
                    cx = Variable(torch.zeros(n, args.latent_dim))
                hidden = (hx, cx)
            else:
                hidden = None
            y_tensor = torch.LongTensor(batch_Y)
            X_dim = batch_X[0][0].shape
            if args.cuda:
                y_GT = Variable(y_tensor.cuda())
            else:
                y_GT = Variable(y_tensor)
            for t in range(T):
                x_tensor = torch.stack([torch.from_numpy(batch_X[sample_id][t]).float() if lengths[sample_id] >= t + 1 
                                            else torch.from_numpy(np.zeros(X_dim)).float() 
                                                for sample_id in range(n)])
                mask = torch.from_numpy(np.array([int(lengths[sample_id] == t + 1) for sample_id in range(n)])).float()
                # y_tensor = torch.stack([torch.from_numpy(batch_Y[sample_id][t]).float() for sample_id in range(n)])
                # print(x_tensor.shape, y_tensor.shape)
                if args.cuda:
                    x_var = Variable(x_tensor.cuda())
                    mask = Variable(mask.cuda())
                else:
                    x_var = Variable(x_tensor)
                    mask = Variable(mask)
                y_pred, hidden = model(x_var, hidden)
                if (t + 1) % args.t_max == 0 and hidden is not None:
                    hidden = (Variable(hidden[0].data), Variable(hidden[1].data))
                pred_loss += (NLL_loss(y_pred, y_GT) * mask).sum(0)    
            pred_loss /= n
            update_network(pred_loss, optimizer)
            print('epoch {} batch {} loss {}'.format(epoch_id, batch_id, pred_loss.data.cpu().numpy()[0]))
            batch_id += 1

        loss = 0
        indices = list(range(N))
        for start in range(0, N, args.batch_size):
            end = min(N - 1, start + args.batch_size - 1)
            batch_X = []
            batch_Y = []
            lengths = []
            for sample_id in indices[start:(end+1)]:
                batch_X.append(X[sample_id])
                batch_Y.append(Y[sample_id])
                lengths.append(len(X[sample_id]))
            # print(batch_X)
            T = max(lengths)
            n = len(batch_X)
            pred_loss = 0
            if args.network_type == 'LSTM':
                if args.cuda:
                    hx = Variable(torch.zeros(n, args.latent_dim).cuda())
                    cx = Variable(torch.zeros(n, args.latent_dim).cuda())
                else:
                    hx = Variable(torch.zeros(n, args.latent_dim))
                    cx = Variable(torch.zeros(n, args.latent_dim))
                hidden = (hx, cx)
            else:
                hidden = None
            y_tensor = torch.LongTensor(batch_Y)
            X_dim = batch_X[0][0].shape
            if args.cuda:
                y_GT = Variable(y_tensor.cuda())
            else:
                y_GT = Variable(y_tensor)
            for t in range(T):
                x_tensor = torch.stack([torch.from_numpy(batch_X[sample_id][t]).float() if lengths[sample_id] >= t + 1 
                                            else torch.from_numpy(np.zeros(X_dim)).float() 
                                                for sample_id in range(n)])
                mask = torch.from_numpy(np.array([int(lengths[sample_id] == t + 1) for sample_id in range(n)])).float()
                # print(t, sum([int(lengths[sample_id] == t + 1) for sample_id in range(n)]))
                # print(x_tensor.shape, y_tensor.shape)
                if args.cuda:
                    x_var = Variable(x_tensor.cuda())
                    mask = Variable(mask.cuda())
                else:
                    x_var = Variable(x_tensor)
                    mask = Variable(mask)
                y_pred, hidden = model(x_var, hidden)
                if (t + 1) % args.t_max == 0 and hidden is not None:
                    hidden = (Variable(hidden[0].data), Variable(hidden[1].data))
                pred_loss += (NLL_loss(y_pred, y_GT) * mask).sum(0)
            loss += pred_loss
        loss /= N
        loss_value = loss.data.cpu().numpy()[0]
        print('loss:', loss_value)
        if loss_value < best_loss - 1e-6 or epoch_id == 0:
            best_loss = loss_value
            save_model(model, checkpoint_dir + "/model_{}".format(args.train_size)) 