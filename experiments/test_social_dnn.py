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
goals = ['leaving', 'blocking']


def prepare_testing_data(trajs, max_episode_length, agent_id, dT=1):
    X = []
    Y = []
    N = len(trajs)
    for episode_id in range(N):
        traj1, traj2 = trajs[episode_id][0], trajs[episode_id][1]
        T = min(len(traj1), args.max_episode_length * dT)
        if agent_id == 0:
            X.append([np.array(phi(traj1[t], traj2[t])) for t in range(0,T-dT,dT)])
        else:
            X.append([np.array(phi(traj2[t], traj1[t])) for t in range(0,T-dT,dT)])
    return X


def test_single_video(X_all, model, args):
    log_prod = []
    for X in X_all:
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
        for t in range(T):
            x_tensor = torch.from_numpy(X[t]).float().unsqueeze(0)
            if args.cuda:
                x_var = Variable(x_tensor.cuda())
            else:
                x_var = Variable(x_tensor)
            y_pred, hidden = model(x_var, hidden)
        log_prod.append(y_pred.data.cpu().numpy()[0][0])
    return log_prod


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
parser.add_argument('--max-episode-length', type=int, default=31, help='Maximum episode length')
parser.add_argument('--network-type', type=str, default='LSTM', help='Network type (MLP, LSTM)')
parser.add_argument('--latent-dim', type=int, default=128, help='Hidden size of LSTM cell')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
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
    
    checkpoint_dir = args.checkpoint_dir + '/{}_{}'.format(args.network_type, goals[0])
    p = Path(checkpoint_dir)
    model0 = DNN(X_DIM, Y_DIM, args.latent_dim, args.network_type, activation='log_softmax')
    if args.cuda: model0.cuda()
    load_model(model0, checkpoint_dir + '/model_{}'.format(args.train_size))
    print(checkpoint_dir)
    model0.eval()

    checkpoint_dir = args.checkpoint_dir + '/{}_{}'.format(args.network_type, goals[1])
    p = Path(checkpoint_dir)
    model1 = DNN(X_DIM, Y_DIM, args.latent_dim, args.network_type, activation='log_softmax')
    if args.cuda: model1.cuda()
    load_model(model1, checkpoint_dir + '/model_{}'.format(args.train_size))
    print(checkpoint_dir)
    model1.eval()


    raw_trajs = dict()
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
        X_all0 = prepare_testing_data(trajs, args.max_episode_length, 0, 1 if i < 13 else 5) # testing entity 0
        X_all1 = prepare_testing_data(trajs, args.max_episode_length, 1, 1 if i < 13 else 5) # testing entity 1

        L00 = test_single_video(X_all0, model0, args)
        L01 = test_single_video(X_all0, model1, args)
        L10 = test_single_video(X_all1, model0, args)
        L11 = test_single_video(X_all1, model1, args)
        # print(np.array([L00, L01, L10, L11]).transpose())

        results[vid_dir] = [L00, L01, L10, L11] 

    pickle.dump(results, 
        open('./data/BaselineDNN_L_{}.pik'.format(args.train_size), 'wb'), 
        protocol=pickle.HIGHEST_PROTOCOL)
