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

from models import GenCoordModel_Physical
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
parser.add_argument('--dataset', type=str, default='collision', help='Dataset name')
parser.add_argument('--max-episode-length', type=int, default=31, help='Maximum episode length')
parser.add_argument('--train-size', type=int, default=50, help='The size of the training set')
parser.add_argument('--rotate-degree', type=float, default=0, help='Coordinate rotation')

if __name__ == '__main__':
    args = parser.parse_args()

    model_dir = args.checkpoint_dir + '/' + args.dataset
    p = Path(model_dir)
    if not p.is_dir():
        p.mkdir(parents=True)

    model_path = str(p / 'model_{}.pik'.format(args.train_size))
    model= GenCoordModel_Physical(model_path=model_path)

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
                x, y = rotate((values[0], values[1]), args.rotate_degree)
                traj1.append((x + 16, y + 12))
                x, y = rotate((values[2], values[3]), args.rotate_degree)
                traj2.append((x + 16, y + 12))
        trajs.append([traj1, traj2])
    data = prepare_physical_data(trajs, dT=1, scale=1.0)

    loss = model.test(data)
    Ds = []
    max_loss = 0.5
    for D, vid_name in zip(loss, vid_names):
        print(vid_name)
        D[0] = [min(d, max_loss) for d in D[0]]
        D[1] = [min(d, max_loss) for d in D[1]]
        print('1:', D[0])
        print('2:', D[1])
        print(len(D[0]))
        print(np.argmax(D[0]), np.argmax(D[1]))
        print(np.mean(D[0]), np.mean(D[1]))
        Ds.append(np.mean(D[0] + D[1]))
    print(Ds)


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
    #     # print(traj2[10:50])
    # data = prepare_physical_data(trajs, dT=3)
    # # print(data['trajs'])
    # # print(data['vels'])
    # loss = model.test(data)
    # Ds = []
    # for D, vid_name in zip(loss, vid_names):
    #     print(vid_name)
    #     # print('1:', D[0])
    #     # print('2:', D[1])
    #     print(len(D[0]))
    #     print(np.argmax(D[0]), np.argmax(D[1]))
    #     print(np.mean(D[0]), np.mean(D[1]))
    #     Ds.append(np.mean(D[0] + D[1]))
    # print(Ds)

    # for i, vid_dir in enumerate(vid_dirs):
    #     print(vid_dir)
    #     trajs = []
    #     vid_id_list = list(range(1, 100, 2)) # only test the videos used as stimuli
    #     for vid_id in vid_id_list:
    #         trajs_file_path = vid_dir + '/{}.txt'.format(vid_id)
    #         traj1, traj2 = [], []
    #         with open(trajs_file_path) as f:
    #             for line in f:
    #                 values = [float(x) for x in line.split()]
    #                 traj1.append((values[0], values[1]))
    #                 traj2.append((values[2], values[3]))
                   
    #         trajs.append([traj1, traj2])
    #     data = prepare_physical_data(trajs, args.max_episode_length, dT=1 if i < 13 else 5)
    #     loss = model.test(data)       
    #     results[vid_dir] = loss 

    # pickle.dump(results, 
    #     open('./data/D_{}_{}.pik'.format(args.dataset, args.train_size), 'wb'), 
    #     protocol=pickle.HIGHEST_PROTOCOL)
        