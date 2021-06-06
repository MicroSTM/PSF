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


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='./record_replay', help='Data directory')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
parser.add_argument('--dataset', type=str, default='blocking_all', help='Dataset name')
parser.add_argument('--num-dirs', type=int, default=4, help='Number of global directions')
parser.add_argument('--trainable-entities', nargs='*', type=int, default=[0, 1], help='Trainable entities')
parser.add_argument('--entity-id', type=int, default=0, help='Which entity to be visualized')
parser.add_argument('--train-size', type=int, default=50, help='The size of the training set')


if __name__ == '__main__':
    args = parser.parse_args()

    model_dir = args.checkpoint_dir + '/' + args.dataset
    p = Path(model_dir)
    if not p.is_dir():
        p.mkdir(parents=True)
    model_path = str(p / 'model_{}.pik'.format(args.train_size))
    model = GenCoordModel_Social(model_path=model_path)
    # model.visualize_force(entity_id=args.entity_id)
    model.visualize_potential(entity_id=args.entity_id, show_ref=args.dataset == 'blocking')
