from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import pickle
import random
import math
import numpy as np
from pathlib import Path

from models import GenCoordModel_Social
from utils import *

goals = ['leaving', 'blocking']


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
parser.add_argument('--max-episode-length', type=int, default=31, help='Maximum episode length')
parser.add_argument('--train-size', type=int, default=50, help='The size of the training set')


if __name__ == '__main__':
    args = parser.parse_args()
    
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

    results = dict()

    model_dir = args.checkpoint_dir
    p = Path(model_dir)

    model_path = str(p / '{}/all_coord_baseline_{}.pik'.format(goals[0], args.train_size))
    model0 = GenCoordModel_Social(model_path=model_path)
    model_path = str(p / '{}/all_coord_baseline_{}.pik'.format(goals[1], args.train_size))
    model1 = GenCoordModel_Social(model_path=model_path)

    for i, vid_dir in enumerate(vid_dirs):
        print(vid_dir)
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
                    if len(traj1) == 1:
                        traj1.append((values[0], values[1]))
                        traj2.append((values[2], values[3]))
            trajs.append([traj1, traj2])
        data01 = prepare_social_data(trajs, args.max_episode_length, dT=1 if i < 13 else 5)

        trajs = []
        vid_id_list = list(range(1, 100, 2))
        for vid_id in vid_id_list:
            trajs_file_path = vid_dir + '/{}.txt'.format(vid_id)
            traj1, traj2 = [], []
            with open(trajs_file_path) as f:
                for line in f:
                    values = [float(x) for x in line.split()]
                    traj1.append((values[0], values[1]))
                    traj2.append((values[2], values[3]))
                    if len(traj1) == 1:
                        traj1.append((values[0], values[1]))
                        traj2.append((values[2], values[3]))
            trajs.append([traj2, traj1])
        data10 = prepare_social_data(trajs, args.max_episode_length, dT=1 if i < 13 else 5)

        print('traj 0, goal 0')
        L00 = model0.test(data01, [0])
        print('traj 0, goal 1')
        L01 = model1.test(data10, [1])
        print('traj 1, goal 0')
        L10 = model0.test(data10, [0])
        print('traj 1, goal 1')
        L11 = model1.test(data01, [1])

        results[vid_dir] = [L00, L01, L10, L11] 

    pickle.dump(results, open(args.data_dir + '/all_coord_baseline_L_{}.pik'.format(args.train_size), 'wb'), 
                    protocol=pickle.HIGHEST_PROTOCOL)
        