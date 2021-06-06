from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import pickle
import random
import math
import numpy as np
from pathlib import Path

from models import GenCoordModel_Physical
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
parser.add_argument('--dataset', type=str, default='collision', help='Dataset name')
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

    model_dir = args.checkpoint_dir + '/' + args.dataset
    p = Path(model_dir)
    if not p.is_dir():
        p.mkdir(parents=True)

    model_path = str(p / 'model_{}.pik'.format(args.train_size))
    model= GenCoordModel_Physical(model_path=model_path)

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
                   
            trajs.append([traj1, traj2])
        data = prepare_physical_data(trajs, args.max_episode_length, dT=1 if i < 13 else 5)
        loss = model.test(data)       
        results[vid_dir] = loss 

    pickle.dump(results, 
        open('./data/D_{}_{}.pik'.format(args.dataset, args.train_size), 'wb'), 
        protocol=pickle.HIGHEST_PROTOCOL)
        