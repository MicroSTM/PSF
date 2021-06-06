from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import pickle
import random
import numpy as np
from pathlib import Path

from models import GenCoordModel_Social


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--data-dir', type=str, default='./data/training_data', help='Data directory')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
parser.add_argument('--dataset', type=str, default='blocking', help='Dataset name')
parser.add_argument('--trainable-entities', nargs='*', type=int, default=[0, 1], help='Trainable entities')
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

    selected_all = []
    coef_all = []

    seed_list = [937, 579, 416, 301, 390, 74, 606, 951, 34, 614]
    for seed in seed_list:
        model = GenCoordModel_Social()
        model.set_seed(seed)
        model.train(data, args.train_size, args.trainable_entities)
        print(model.res['selected'], model.res['reg'].coef_)
        selected_all.append(model.res['selected'])
        coef_all.append(model.res['reg'].coef_)

    print(selected_all)
    print(coef_all)
    pickle.dump({'selected_all': selected_all, 'coef_all': coef_all}, open('{}/10runs.pik'.format(args.checkpoint_dir + '/' + args.dataset), 'wb'))

