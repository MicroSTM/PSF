from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import pickle
import math
import numpy as np
from pathlib import Path
from utils import *


MASS = math.pi
cx, cy = 16, 12 # center position
half_x, half_y, door_length = 8, 8, 3


def prepare_dataset(trajs_raw, dT=5):
    trajs, vels, accs, Fs, segs = [], [], [], [], []
    for episode in trajs_raw:
        traj1, traj2 = episode[0], episode[1]
        T = len(traj1)
        vels1, vels2 = [], []
        for t in range(0, T - dT, dT):
            vel1 = rescale((traj1[t + 1][0] - traj1[t][0], traj1[t + 1][1] - traj1[t][1]), dT)
            vel2 = rescale((traj2[t + 1][0] - traj2[t][0], traj2[t + 1][1] - traj2[t][1]), dT)
            vels1.append(vel1)
            vels2.append(vel2)
        trajs.append([traj1[:T-dT:dT], traj2[:T-dT:dT]])
        vels.append([vels1, vels2])
        T = len(vels1)
        accs1, accs2 = [], []
        for t in range(0, T - 1):
            acc1 = (vels1[t + 1][0] - vels1[t][0], vels1[t + 1][1] - vels1[t][1])
            acc2 = (vels2[t + 1][0] - vels2[t][0], vels2[t + 1][1] - vels2[t][1])
            accs1.append(acc1)
            accs2.append(acc2)
        accs.append([accs1, accs2])
        Fs1 = [rescale(a, MASS) for a in accs1]
        Fs2 = [rescale(a, MASS) for a in accs2]
        Fs.append([Fs1, Fs2])
        print(len(trajs[-1][0]), len(vels1), len(accs1), len(Fs1))
        segs.append([
            [(cx - half_x, cy + half_y), (cx + half_x, cy + half_y)],
            [(cx + half_x, cy + half_y), (cx + half_x, cy - half_y)],
            [(cx + half_x, cy - half_y), (cx - half_x, cy - half_y)],
            [(cx - half_x, cy - half_y), (cx - half_x, cy + half_y - door_length)]
            ])

    return {'trajs': trajs, 'vels': vels, 'segs': segs, 'accs': accs, 'Fs': Fs}


parser = argparse.ArgumentParser()
parser.add_argument('--record-dir', type=str, default='./record_replay', metavar='RECORD', help='Mission record directory')
parser.add_argument('--data-dir', type=str, default='/home/stm/MultiAgent/HeiderSimmel/data', metavar='RECORD', help='Mission record directory')
parser.add_argument('--dataset', type=str, default='collision_same', help='Dataset name')


if __name__ == '__main__':
    args = parser.parse_args()

    record_dir = args.record_dir
    p = Path(record_dir) / '{}'.format(args.dataset)
    if not p.is_dir():
        p.mkdir(parents = True)

    trajs = pickle.load(open(args.data_dir + '/{}.pik'.format(args.dataset), 'rb'))
    data = prepare_dataset(trajs[:500])
    pickle.dump(data, open(str(p / 'data.pik'), 'wb'))
