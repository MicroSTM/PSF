from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import pickle
import random
import numpy as np
from pathlib import Path

from models import GenCoordModel_Physical


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--data-dir', type=str, default='./data/training_data', help='Data directory')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
parser.add_argument('--dataset', type=str, default='collision', help='Dataset name')
parser.add_argument('--train-size', type=int, default=50, help='The size of the training set')


if __name__ == '__main__':
    args = parser.parse_args()
    print (' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    data_path = args.data_dir + '/' + args.dataset + '/data.pik'
    data = pickle.load(open(data_path, 'rb'))

    model_dir = args.checkpoint_dir + '/' + args.dataset
    p = Path(model_dir)
    if not p.is_dir():
        p.mkdir(parents=True)
    model_path = str(p / 'model_{}.pik'.format(args.train_size))

    random.seed(args.seed)
    model = GenCoordModel_Physical(model_path=model_path)
    model.train(data, args.train_size)