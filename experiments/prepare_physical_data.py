from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import random
import math
import sys
import pickle
from pathlib import Path

from utils import *


MASS = math.pi
cx, cy = 16, 12
half_x, half_y, door_length = 8, 8, 3


def smoothListGaussian(list_, degree=5):  

    window = degree * 2 - 1 
    weight = np.array([1.0] * window)  
    weightGauss=[]  
    for i in range(window):  
        i = i - degree + 1  
        frac = i / float(window)  
        gauss = 1 / (np.exp((4*(frac))**2))  
        weightGauss.append(gauss)  

    weight = np.array(weightGauss) * weight  
    smoothed=[0.0]*(len(list_) - window)  

    for i in range(len(smoothed)):  
        smoothed[i]=sum(np.array(list_[i:i+window])*weight)/sum(weight)  

    return smoothed 


def smooth_traj(traj, dT=1):
    smoothed_traj = list(traj)
    T = len(traj)
    for t in range(dT, T - dT, 1):
        pos = (sum([traj[tt][0] for tt in range(t - dT, t + dT + 1)]) / (dT * 2 + 1),
               sum([traj[tt][1] for tt in range(t - dT, t + dT + 1)]) / (dT * 2 + 1))
        smoothed_traj[t] = pos
    return smoothed_traj


def prepare_dataset(trajs_raw, dT=1, remove_last=False):
    trajs, vels, accs, Fs, segs = [], [], [], [], []
    for episode in trajs_raw:
        # print(episode[1])
        traj1, traj2 = smooth_traj(episode[0], dT=1 if dT == 1 else 0), \
                       smooth_traj(episode[1], dT=1 if dT == 1 else 0)
        # print(traj2)
        T = len(traj1)
        vels1, vels2 = [], []
        max_T = T - dT * 2 if remove_last else T - dT
        for t in range(0, max_T, dT):
            vel1 = rescale((traj1[t + 1][0] - traj1[t][0], traj1[t + 1][1] - traj1[t][1]), dT)
            vel2 = rescale((traj2[t + 1][0] - traj2[t][0], traj2[t + 1][1] - traj2[t][1]), dT)
            vels1.append(vel1)
            vels2.append(vel2)
        trajs.append([traj1[:max_T:dT], traj2[:max_T:dT]])
        # print(trajs[-1][0][-1])
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

parser.add_argument('--record-dir', type=str, default='./data/training_data', help='Mission record directory')
parser.add_argument('--dataset', type=str, default='physical', help='Dataset name')
parser.add_argument('--source', type=int, default=0, help='Which version of data')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.source == 0:
        vid_dirs = [
            './data/animations_all/OO/' + args.dataset,
            # './data/animations_all/OO/spring',
        ]

    record_dir = args.record_dir
    p = Path(record_dir) / args.dataset
    if not p.is_dir():
        p.mkdir(parents = True)

    trajs = []
    for i, vid_dir in enumerate(vid_dirs):
        # trajs = []
        labels = []
        # for vid_id in range(1, 100, 2):
        vid_id_list = list(range(0, 100, 2)) if args.source == 0 else list(range(100))
        cnt = 0
        for vid_id in vid_id_list:
            trajs_file_path = vid_dir + '/{}.txt'.format(vid_id)
            traj1, traj2 = [], []
            with open(trajs_file_path) as f:
                for line in f:
                    values = [float(x) for x in line.split()]
                    traj1.append((values[0], values[1]))
                    traj2.append((values[2], values[3]))
                    # if len(traj1) == 1:
                    #     traj1.append((values[0], values[1]))
                    #     traj2.append((values[2], values[3]))
            # print(dist(traj2[0], init_pos[0]))
            trajs.append([traj1, traj2])
        print(vid_dir)

    data = prepare_dataset(trajs, 
                           dT=5,
                           remove_last=False)#args.source != 0 and args.source != -3)
    pickle.dump(data, open(str(p / 'data.pik'), 'wb'))
    print('Number of videos:', len(trajs))