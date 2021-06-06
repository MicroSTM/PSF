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
Y_DIM = 4


def prepare_testing_data(trajs, max_episode_length, dT):
    X = []
    Y = []
    N = len(trajs)
    for episode_id in range(N):
        max_len = min(dT * max_episode_length, len(trajs[episode_id][0]))
        traj1, traj2 = trajs[episode_id][0][:max_len], trajs[episode_id][1][:max_len]
        T = max_len
        vel1, vel2 = [], []
        for t in range(0, T - dT, dT):
            v1 = rescale((traj1[t + 1][0] - traj1[t][0], traj1[t + 1][1] - traj1[t][1]), dT)
            v2 = rescale((traj2[t + 1][0] - traj2[t][0], traj2[t + 1][1] - traj2[t][1]), dT)
            vel1.append(v1)
            vel2.append(v2)

        T = len(vel1)
        cur_x, cur_y = [], []
        # for t in range(T - 1):
        #     feat_cur = phi(traj1[t * dT], traj2[t * dT])
        #     feat_nxt = phi(traj1[(t + 1) * dT], traj2[(t + 1) * dT])
        #     feat = feat_cur + feat_nxt
        #     vel = [vel1[t + 1][0], vel1[t + 1][1], vel2[t + 1][0], vel2[t + 1][1]]
        #     x = np.array(feat)
        #     y = np.array(vel)
        #     cur_x.append(x)
        #     cur_y.append(y)
        for t in range(T):
            feat_cur = phi(traj1[t * dT], traj2[t * dT])
            feat = feat_cur
            vel = [vel1[t][0], vel1[t][1], vel2[t][0], vel2[t][1]]
            x = np.array(feat)
            y = np.array(vel)
            cur_x.append(x)
            cur_y.append(y)
        X.append(cur_x)
        Y.append(cur_y)
    return X, Y


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
parser.add_argument('--max-episode-length', type=int, default=31, help='Maximum episode length')
parser.add_argument('--network-type', type=str, default='LSTM', help='Network type (MLP, LSTM)')
parser.add_argument('--latent-dim', type=int, default=128, help='Hidden size of LSTM cell')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
parser.add_argument('--dataset', type=str, default='collision', help='Dataset name')
parser.add_argument('--train-size', type=int, default=50, help='The size of the training set')


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    vid_dirs = [
    args.data_dir + '/animations_all/HH/Style1/7',
    args.data_dir + '/animations_all/HH/Style1/10',
    args.data_dir + '/animations_all/HH/Style1/20',
    args.data_dir + '/animations_all/HH/Style1/50',
    args.data_dir + '/animations_all/HH/Style1/100',
    args.data_dir + '/animations_all/HH/Style2/7',
    args.data_dir + '/animations_all/HH/Style2/10',
    args.data_dir + '/animations_all/HH/Style2/20',
    args.data_dir + '/animations_all/HH/Style2/50',
    args.data_dir + '/animations_all/HH/Style2/100',
    args.data_dir + '/animations_all/HO/Style1',
    args.data_dir + '/animations_all/HO/Style2',
    args.data_dir + '/animations_all/HO/Style3',
    args.data_dir + '/animations_all/OO/collision',
    args.data_dir + '/animations_all/OO/rod',
    args.data_dir + '/animations_all/OO/rope',
    args.data_dir + '/animations_all/OO/spring',
    ]
    
    checkpoint_dir = args.checkpoint_dir + '/{}_{}'.format(args.network_type, args.dataset)
    p = Path(checkpoint_dir)
    if not p.is_dir():
        p.mkdir(parents = True)

    model = DNN(X_DIM, Y_DIM, args.latent_dim, args.network_type)
    if args.cuda:
        model.cuda()
        MSE_loss = torch.nn.MSELoss().cuda()
    else:
        MSE_loss = torch.nn.MSELoss()
    load_model(model, checkpoint_dir + '/model_{}'.format(args.train_size))
    print(checkpoint_dir)
    model.eval()

    raw_trajs = dict()
    vels = dict()
    results = dict()

    for i, vid_dir in enumerate(vid_dirs):
        print(vid_dir)
        loss = []
        trajs = []
        vid_id_list = list(range(1, 100, 2)) # only test the videos used as stimuli
        for vid_id in vid_id_list:
            trajs_file_path = vid_dir + '/{}.txt'.format(vid_id)
            traj1, traj2 = [], []
            with open(trajs_file_path) as f:
                for line in f:
                    values = [float(x) for x in line.split()]
                    traj1.append((values[0], values[1]))
                    traj2.append((values[2], values[3]))
            trajs.append([traj1, traj2])
        X_all, Y_all = prepare_testing_data(trajs, args.max_episode_length, 1 if i < 13 else 5)
        for X, Y in zip(X_all, Y_all):
            T = len(X)
            if args.network_type == 'LSTM':
                if args.cuda:
                    hx = Variable(torch.zeros(1, args.latent_dim).cuda())
                    cx = Variable(torch.zeros(1, args.latent_dim).cuda())
                else:
                    hx = Variable(torch.zeros(1, args.latent_dim))
                    cx = Variable(torch.zeros(1, args.latent_dim))
                hidden = (hx, cx)
            else:
                hidden = None
            mse_loss_1, mse_loss_2 = [], []
            for t in range(T):
                x_tensor = torch.from_numpy(X[t]).float().unsqueeze(0)
                y_tensor = torch.from_numpy(Y[t]).float().unsqueeze(0)
                if args.cuda:
                    x_var = Variable(x_tensor.cuda())
                    y_GT = Variable(y_tensor.cuda())
                else:
                    x_var = Variable(x_tensor)
                    y_GT = Variable(y_tensor)
                y_pred, hidden = model(x_var, hidden)
                if t: # do not evaluat the first time step (not enough info)
                    mse_loss_1.append(float(MSE_loss(y_pred[:, 0:2], y_GT[:, 0:2]).data.cpu().numpy()[0]))
                    mse_loss_2.append(float(MSE_loss(y_pred[:, 2:4], y_GT[:, 2:4]).data.cpu().numpy()[0]))
            loss.append([mse_loss_1, mse_loss_2])
        results[vid_dir] = loss 

    pickle.dump(results, 
        open('./data/BaselineDNN_D_{}_{}.pik'.format(args.dataset, args.train_size), 'wb'), 
        protocol=pickle.HIGHEST_PROTOCOL)
