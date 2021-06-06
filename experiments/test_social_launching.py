from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import pickle
import random
import math
import numpy as np
from pathlib import Path
from scipy.io import loadmat

from models import GenCoordModel_Social
from utils import *

goals = ['leaving', 'blocking']


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
parser.add_argument('--max-episode-length', type=int, default=31, help='Maximum episode length')
parser.add_argument('--train-size', type=int, default=50, help='The size of the training set')
parser.add_argument('--rotate-degree', type=float, default=0, help='Coordinate rotation')
parser.add_argument('--shift-x', type=float, default=0, help='X axis shift')
parser.add_argument('--shift-y', type=float, default=0, help='Y axis shift')


if __name__ == '__main__':
    args = parser.parse_args()

    model_dir = args.checkpoint_dir
    p = Path(model_dir)

    model_path = str(p / '{}/model_{}.pik'.format(goals[0], args.train_size))
    model0 = GenCoordModel_Social(model_path=model_path)
    model_path = str(p / '{}/model_{}.pik'.format(goals[1], args.train_size))
    model1 = GenCoordModel_Social(model_path=model_path)

    vid_names = [
        'launching',
        'entraining',
        'tempgap',
        'triggering',
        'spatialgap'
    ]

    scales = {
        'launching': 2.43,
        'entraining': 2.22,#1.96,
        'tempgap': 2.45,
        'triggering': 2.35,
        'spatialgap': 2.22
    }

    trajs = []
    for i, vid_name in enumerate(vid_names):
        trajs_file_path = args.data_dir + '/animations_all/launching/{}.txt'.format(vid_name)
        traj1, traj2 = [], []
        scale = scales[vid_name]
        with open(trajs_file_path) as f:
            for line in f:
                values = [float(x) * scale for x in line.split()]
                x, y = rotate((values[0] + 1, values[1]), args.rotate_degree)
                traj1.append((x + 16 + args.shift_x, y + 12 + args.shift_y))
                x, y = rotate((values[2] + 1, values[3]), args.rotate_degree)
                traj2.append((x + 16 + args.shift_x, y + 12 + args.shift_y))
        print(traj1)
        trajs.append([traj1, traj2])
    data01 = prepare_social_data(trajs, dT=1)
    trajs = []
    for i, vid_name in enumerate(vid_names):
        trajs_file_path = args.data_dir + '/animations_all/launching/{}.txt'.format(vid_name)
        traj1, traj2 = [], []
        scale = scales[vid_name]
        with open(trajs_file_path) as f:
            for line in f:
                values = [float(x) * scale for x in line.split()]
                x, y = rotate((values[0] + 1, values[1]), args.rotate_degree)
                traj1.append((x + 16 + args.shift_x, y + 12 + args.shift_y))
                x, y = rotate((values[2] + 1, values[3]), args.rotate_degree)
                traj2.append((x + 16 + args.shift_x, y + 12 + args.shift_y))
        trajs.append([traj2, traj1])
    data10 = prepare_social_data(trajs, dT=1)
    # for i, vid_name in enumerate(vid_names):
    #     trajs_file_path = args.data_dir + '/animations_all/launching/{}.mat'.format(vid_name)
    #     data = loadmat(trajs_file_path)
    #     traj1, traj2 = [], []
    #     T = len(data['x_b1'][0])
    #     for t in range(T):
    #         x, y = rotate((data['x_b1'][0][t], data['y_b1'][0][t]), args.rotate_degree)
    #         traj1.append((x + 16, y + 12))
    #         x, y = rotate((data['x_b2'][0][t], data['y_b2'][0][t]), args.rotate_degree)
    #         traj2.append((x + 16, y + 12))
    #     if T > 50:
    #         trajs.append([traj1[5:50], traj2[5:50]])
    #     else:
    #         trajs.append([traj1[5:], traj2[5:]])
    #     # print(traj1)
    #     # print(traj2)
    # data01 = prepare_social_data(trajs, dT=3)
    # trajs = []
    # for i, vid_name in enumerate(vid_names):
    #     trajs_file_path = args.data_dir + '/animations_all/launching/{}.mat'.format(vid_name)
    #     data = loadmat(trajs_file_path)
    #     traj1, traj2 = [], []
    #     T = len(data['x_b1'][0])
    #     for t in range(T):
    #         x, y = rotate((data['x_b1'][0][t], data['y_b1'][0][t]), args.rotate_degree)
    #         traj1.append((x + 16, y + 12))
    #         x, y = rotate((data['x_b2'][0][t], data['y_b2'][0][t]), args.rotate_degree)
    #         traj2.append((x + 16, y + 12))
    #     if T > 50:
    #         trajs.append([traj1[5:50], traj2[5:50]])
    #     else:
    #         trajs.append([traj1[5:], traj2[5:]])
    # data10 = prepare_social_data(trajs, dT=3)

    print('traj 0, goal 0')
    L00 = model0.test(data01, [0])
    print('traj 0, goal 1')
    L01 = model1.test(data10, [1])
    print('traj 1, goal 0')
    L10 = model0.test(data10, [0])
    print('traj 1, goal 1')
    L11 = model1.test(data01, [1])

    Ls = []

    for i, vid_name in enumerate(vid_names):
        print(vid_name)
        print((max(np.mean(L00[i]), np.mean(L01[i])) + max(np.mean(L10[i]), np.mean(L11[i]))) * 0.5)
        print(np.mean(L00[i]), np.mean(L01[i]), np.mean(L10[i]), np.mean(L11[i]))
        Ls.append((max(np.mean(L00[i]), np.mean(L01[i])) + max(np.mean(L10[i]), np.mean(L11[i]))) * 0.5)
    print(Ls)

    # Ls = [0] * len(vid_names)
    # for rotate_degree_id in range(8):
    #     rotate_degree = rotate_degree_id * 0.25
    #     trajs = []
    #     for i, vid_name in enumerate(vid_names):
    #         trajs_file_path = args.data_dir + '/animations_all/launching/{}.mat'.format(vid_name)
    #         data = loadmat(trajs_file_path)
    #         traj1, traj2 = [], []
    #         T = len(data['x_b1'][0])
    #         for t in range(T):
    #             x, y = rotate((data['x_b1'][0][t], data['y_b1'][0][t]), rotate_degree)
    #             traj1.append((x + 16, y + 12))
    #             x, y = rotate((data['x_b2'][0][t], data['y_b2'][0][t]), rotate_degree)
    #             traj2.append((x + 16, y + 12))
    #         if T > 50:
    #             trajs.append([traj1[5:50], traj2[5:50]])
    #         else:
    #             trajs.append([traj1[5:], traj2[5:]])
    #         # print(traj1)
    #         # print(traj2)
    #     data01 = prepare_social_data(trajs, dT=3)
    #     trajs = []
    #     for i, vid_name in enumerate(vid_names):
    #         trajs_file_path = args.data_dir + '/animations_all/launching/{}.mat'.format(vid_name)
    #         data = loadmat(trajs_file_path)
    #         traj1, traj2 = [], []
    #         T = len(data['x_b1'][0])
    #         for t in range(T):
    #             x, y = rotate((data['x_b1'][0][t], data['y_b1'][0][t]), rotate_degree)
    #             traj1.append((x + 16, y + 12))
    #             x, y = rotate((data['x_b2'][0][t], data['y_b2'][0][t]), rotate_degree)
    #             traj2.append((x + 16, y + 12))
    #         if T > 50:
    #             trajs.append([traj1[5:50], traj2[5:50]])
    #         else:
    #             trajs.append([traj1[5:], traj2[5:]])
    #     data10 = prepare_social_data(trajs, dT=3)

    #     L00 = model0.test(data01, [0])
    #     L01 = model1.test(data10, [1])
    #     L10 = model0.test(data10, [0])
    #     L11 = model1.test(data01, [1])

    #     for i, vid_name in enumerate(vid_names):
    #         Ls[i] = ((max(np.mean(L00[i]), np.mean(L01[i])) + max(np.mean(L10[i]), np.mean(L11[i]))) * 0.5) / 8
    # print(Ls)


