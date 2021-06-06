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

from models import PredMLP

X_DIM = 28
Y_DIM = 4

def phi(x1, x2):
    points = [(-8, 5), (-8, -8), (8, -8), (8, 8), (-8, 8)]
    p1 = (x1[0] - 16, x1[1] - 12)
    p2 = (x2[0] - 16, x2[1] - 12)
    feat = [p1[0], p1[1]]
    feat += [p1[0] - p2[0], p1[1] - p2[1]]
    for point in points:
        feat += [p1[0] - point[0], p1[1] - point[1]]
    feat += [p2[0], p2[1]]
    feat += [p2[0] - p1[0], p2[1] - p1[1]]
    for point in points:
        feat += [p2[0] - point[0], p2[1] - point[1]]
    return feat


def prepare_dataset(trajs, dT=5):
    X = []
    Y = []
    for episode in trajs:
        traj1, traj2 = episode[0], episode[1]
        T = len(traj1)
        t = 0
        cur_x, cur_y = [], []
        while t + dT < T:
            feat = phi(traj1[t], traj2[t])
            vel = [traj1[t + dT][0] - traj1[t][0], traj1[t + dT][1] - traj1[t][1], 
                   traj2[t + dT][0] - traj2[t][0], traj2[t + dT][1] - traj2[t][1]]
            t += dT
            x = np.array(feat)
            y = np.array(vel)
            cur_x.append(x)
            cur_y.append(y)
        X.append(cur_x)
        Y.append(cur_y)
    return X, Y


def _update_network(loss, optimizer):
    """update network parameters"""
    optimizer.zero_grad()
    loss.backward()
    # nn.utils.clip_grad_norm(list(hist_model.parameters()) + list(ind_model.parameters()) + list(global_model.parameters()), 
    #                         args.max_gradient_norm, 1)
    optimizer.step()


def save_model(model, path):
    """save trained model parameters"""
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """load trained model parameters"""
    model.load_state_dict(dict(torch.load(path)))


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--t-max', type=int, default=50, metavar='STEPS', help='Max number of forward steps for A2C before update')
parser.add_argument('--max-episode-length', type=int, default=30, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--latent-dim', type=int, default=128, metavar='SIZE', help='Hidden size of LSTM cell')
parser.add_argument('--lr', type=float, default=3e-4, metavar='η', help='Learning rate')
parser.add_argument('--batch-size', type=int, default=64, metavar='SIZE', help='Off-policy batch size')
parser.add_argument('--checkpoint-dir', type=str, default='/home/stm/MultiAgent/HeiderSimmel/checkpoints', metavar='RECORD', help='Mission record directory')
parser.add_argument('--record-dir', type=str, default='/home/stm/MultiAgent/HeiderSimmel/record_replay', metavar='RECORD', help='Mission record directory')
parser.add_argument('--data-dir', type=str, default='/home/stm/MultiAgent/HeiderSimmel/data', metavar='RECORD', help='Mission record directory')
parser.add_argument('--image-height', type=str, default=84, metavar='IMAGE_SIZE', help='The height of the input image')
parser.add_argument('--image-width', type=str, default=84, metavar='IMAGE_SIZE', help='The width of the input image')
parser.add_argument('--image-channels', type=str, default=3, metavar='IMAGE_SIZE', help='The number of channels of the input image')
parser.add_argument('--checkpoint-epoches', type=int, default=1, metavar='CHECKPOINTS', help='Frequency of saving checkpoints')
parser.add_argument('--train_ratio', type=float, default=0.75, metavar='RATIO', help='Ratio of training instances')
parser.add_argument('--dropout-rate', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--dataset', type=str, default='collision_same', help='Dataset name')
parser.add_argument('--network', type=str, default='MLP', help='Network type')

vid_dirs = [
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style1/7',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style1/10',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style1/20',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style1/50',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style1/100',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style2/7',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style2/10',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style2/20',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style2/50',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style2/100',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style3',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style4',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HO/Style1',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HO/Style2',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/OO/collision/same_density',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/OO/collision/different_densities',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/OO/rod',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/OO/rope',
'/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/OO/spring',
]


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print (' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    checkpoint_dir = args.checkpoint_dir + "_" + args.dataset + "_" + args.network
    p = Path(checkpoint_dir)
    if not p.is_dir():
        p.mkdir(parents = True)

    model = PredMLP(X_DIM, Y_DIM, args.latent_dim)
    if args.cuda:
        model.cuda()
        MSE_loss = torch.nn.MSELoss().cuda()
    else:
        MSE_loss = torch.nn.MSELoss()
    load_model(model, checkpoint_dir + '/model')
    model.eval()

    pred_results = dict()
    raw_trajs = dict()

    for vid_dir in vid_dirs:
        trajs = []
        for vid_id in range(1, 100, 2):
            trajs_file_path = vid_dir + '/{}.txt'.format(vid_id)
            traj1, traj2 = [], []
            with open(trajs_file_path) as f:
                for line in f:
                    values = [float(x) for x in line.split()]
                    traj1.append((values[0], values[1]))
                    traj2.append((values[2], values[3]))
            trajs.append([traj1, traj2])
            # print(trajs)
        X_all, Y_all = prepare_dataset(trajs)
        # print(X_all)
        pred_results[vid_dir] = dict()
        raw_trajs[vid_dir] = trajs
        for vid_id, X, Y in zip(range(1, 100, 2), X_all, Y_all):
            T = len(X)
            if args.cuda:
                hx = Variable(torch.zeros(1, args.latent_dim).cuda())
                cx = Variable(torch.zeros(1, args.latent_dim).cuda())
            else:
                hx = Variable(torch.zeros(1, args.latent_dim))
                cx = Variable(torch.zeros(1, args.latent_dim))
            mse_loss_1, mse_loss_2 = [], []
            for t in range(T):
                x_tensor = torch.from_numpy(X[t]).float().unsqueeze(0)
                y_tensor = torch.from_numpy(Y[t]).float().unsqueeze(0)
                # print(x_tensor.shape, y_tensor.shape)
                if args.cuda:
                    x_var = Variable(x_tensor.cuda())
                    y_GT = Variable(y_tensor.cuda())
                else:
                    x_var = Variable(x_tensor)
                    y_GT = Variable(y_tensor)
                y_pred, (hx, cx) = model(x_var, (hx, cx))
                mse_loss_1.append(MSE_loss(y_pred[:, 0:2], y_GT[:, 0:2]))
                mse_loss_2.append(MSE_loss(y_pred[:, 2:4], y_GT[:, 2:4]))
            pred_results[vid_dir][vid_id] = [mse_loss_1, mse_loss_2]
    pickle.dump({'loss': pred_results, 'trajs': raw_trajs}, 
               open(args.record_dir + '/pred_results_{}.pik'.format(args.dataset), 'wb'), 
               protocol=pickle.HIGHEST_PROTOCOL)


    # trajs = pickle.load(open(args.data_dir + '/{}.pik'.format(args.dataset), 'rb'))
    # X, Y = prepare_dataset(trajs)
    # N = len(X)
    # N_train = int(N * args.train_ratio)
    # N_test = N - N_train
    # X_train, Y_train = X[:N_train], Y[:N_train]
    # X_test, Y_test = X[N_train:], Y[N_train:]

    

    # nb_epoch = 100
    # best_val_loss = 1000
    # for epoch_id in range(nb_epoch):
    #     indices = list(range(N_train))
    #     random.shuffle(indices)
    #     batch_id = 0
    #     for start in range(0, N_train, args.batch_size):
    #         end = min(N_train - 1, start + args.batch_size - 1)
    #         batch_X = []
    #         batch_Y = []
    #         for sample_id in indices[start:(end+1)]:
    #             batch_X.append(X_train[sample_id])
    #             batch_Y.append(Y_train[sample_id])
    #         # print(batch_X)
    #         T = len(batch_X[0])
    #         n = len(batch_X)
    #         mse_loss = 0
    #         if args.cuda:
    #             hx = Variable(torch.zeros(n, args.latent_dim).cuda())
    #             cx = Variable(torch.zeros(n, args.latent_dim).cuda())
    #         else:
    #             hx = Variable(torch.zeros(n, args.latent_dim))
    #             cx = Variable(torch.zeros(n, args.latent_dim))
    #         for t in range(T):
    #             x_tensor = torch.stack([torch.from_numpy(batch_X[sample_id][t]).float() for sample_id in range(n)])
    #             y_tensor = torch.stack([torch.from_numpy(batch_Y[sample_id][t]).float() for sample_id in range(n)])
    #             # print(x_tensor.shape, y_tensor.shape)
    #             if args.cuda:
    #                 x_var = Variable(x_tensor.cuda())
    #                 y_GT = Variable(y_tensor.cuda())
    #             else:
    #                 x_var = Variable(x_tensor)
    #                 y_GT = Variable(y_tensor)
    #             y_pred, (hx, cx) = model(x_var, (hx, cx))
    #             if (t + 1) % args.t_max == 0:
    #                 hx = Variable(hx.data)
    #                 cx = Variable(cx.data)
    #             mse_loss += MSE_loss(y_pred, y_GT)
    #         mse_loss /= T
    #         _update_network(mse_loss, optimizer)
    #         print('epoch {} batch {} loss {}'.format(epoch_id, batch_id, mse_loss.data.cpu().numpy()[0]))
    #         batch_id += 1

    #     val_loss = 0
    #     indices = list(range(N_test))
    #     for start in range(0, N_test, args.batch_size):
    #         end = min(N_train - 1, start + args.batch_size - 1)
    #         batch_X = []
    #         batch_Y = []
    #         for sample_id in indices[start:(end+1)]:
    #             batch_X.append(X_test[sample_id])
    #             batch_Y.append(Y_test[sample_id])
    #         # print(batch_X)
    #         T = len(batch_X[0])
    #         n = len(batch_X)
    #         mse_loss = 0
    #         if args.cuda:
    #             hx = Variable(torch.zeros(n, args.latent_dim).cuda())
    #             cx = Variable(torch.zeros(n, args.latent_dim).cuda())
    #         else:
    #             hx = Variable(torch.zeros(n, args.latent_dim))
    #             cx = Variable(torch.zeros(n, args.latent_dim))
    #         for t in range(T):
    #             x_tensor = torch.stack([torch.from_numpy(batch_X[sample_id][t]).float() for sample_id in range(n)])
    #             y_tensor = torch.stack([torch.from_numpy(batch_Y[sample_id][t]).float() for sample_id in range(n)])
    #             # print(x_tensor.shape, y_tensor.shape)
    #             if args.cuda:
    #                 x_var = Variable(x_tensor.cuda())
    #                 y_GT = Variable(y_tensor.cuda())
    #             else:
    #                 x_var = Variable(x_tensor)
    #                 y_GT = Variable(y_tensor)
    #             y_pred, (hx, cx) = model(x_var, (hx, cx))
    #             if (t + 1) % args.t_max == 0:
    #                 hx = Variable(hx.data)
    #                 cx = Variable(cx.data)
    #             mse_loss += MSE_loss(y_pred, y_GT) * n
    #         mse_loss /= T 
    #         val_loss += mse_loss
    #     val_loss /= N_test
    #     val_loss_value = val_loss.data.cpu().numpy()[0]
    #     print('val_loss:', val_loss_value)
    #     if val_loss_value < best_val_loss - 1e-6 or epoch_id == 0:
    #         best_val_loss = val_loss_value
    #         save_model(model, checkpoint_dir + "/model")

    #     # batch = trajs[]
    # 