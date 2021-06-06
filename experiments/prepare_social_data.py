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
    episode_indices = list(range(len(trajs_raw)))
    random.shuffle(episode_indices)
    for episode_id in episode_indices:
        episode = trajs_raw[episode_id]
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
parser.add_argument('--data-dir', type=str, default='./data/training_data', help='Mission record directory')
parser.add_argument('--dataset', type=str, default='blocking', help='Dataset name')
parser.add_argument('--init-cond', type=int, default=0, help='Initial condition')
parser.add_argument('--max-length', type=int, default=31, help='Maximum episode length')
parser.add_argument('--min-length', type=int, default=0, help='Maximum episode length')
parser.add_argument('--source', type=int, default=0, help='Which version of data')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.source == 0:
        vid_dirs = [
        # # '/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style1/7',
        # # '/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style1/10',
        # # '/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style1/20',
        # # '/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style1/50',
        '/home/stm/Dropbox/MultiAgentCode/HeiderSimmel/record_replay/Blocking_v1/2000_16000/100',
        # '/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style1/100',
        # # '/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style2/7',
        # # '/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style2/10',
        # # '/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style2/20',
        # # '/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style2/50',
        '/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style2/100',
        # '/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style3',
        # '/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HH/Style4',
        # '/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HO/Style1',
        # '/home/stm/Dropbox/Viz_HeiderSimmel/ForHumanExperiments/New/HO/Style2',
        # '/home/stm/MultiAgent/HeiderSimmel/record_replay/Blocking_v1/1_1/0/0_0/100',
        ]
    elif args.source == 1:
        vid_dirs = [
        '/home/stm/MultiAgent/HeiderSimmel/record_replay/Blocking_v1/off_1_1/1/12000_12000/100'
        ]
    elif args.source == 2:
        vid_dirs = [
        # '/home/stm/HSGenRL/record/Blocking_v5/S11_S21_MLP_L30_B0_E0.0/3/90000_90000/100'
        '/home/stm/HSGenRL/record/Blocking_v5/S11_S21_MLP_L30_B0_E0.0/1/160000_160000/100'
        ]
    else:
        vid_dirs = [
        # '/home/stm/Dropbox/MultiAgentCode/HeiderSimmel/record_replay/Blocking_v1/50000_1/100'
        # '/home/stm/Dropbox/MultiAgentCode/HeiderSimmel/record_replay/Blocking_v1/120000/100',
        '/home/stm/Dropbox/MultiAgentCode/HeiderSimmel/record_replay/Blocking_v1/400000/100',
        '/home/stm/Dropbox/MultiAgentCode/HeiderSimmel/record_replay/Blocking_v1/370000/100',
        ]        


    data_dir = args.data_dir
    p = Path(data_dir) / '{}'.format(args.dataset)
    if not p.is_dir():
        p.mkdir(parents = True)


    # init_pos = [(10, 6), (10, 18), (22, 18), (22, 6)]
    init_pos = [(9, 5), (23, 19), (23, 5)]
    cnt_init_cond = [0] * len(init_pos)

    trajs = []
    for i, vid_dir in enumerate(vid_dirs):
        # trajs = []
        labels = []
        # for vid_id in range(1, 100, 2):
        vid_id_list = list(range(0, 120 if args.source == 0 and i == 0 else 100, 2)) if args.source == 0 or args.source == 2 else list(range(100))
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
            if args.init_cond >= 0 and dist(traj2[0], init_pos[args.init_cond]) > 1e-6: continue
            T = min(args.max_length, len(traj1))
            max_length = [13, 11, 15]
            if len(traj1) > args.max_length or len(traj1) < args.min_length: continue
            # if traj1[-1][0] >= 7: continue
            print(vid_id)
            if args.source == 3:
                exceed = False
                for cond_id, init_cond in enumerate(init_pos):
                    if dist(traj1[0], init_cond) < 1e-6:
                        if cnt_init_cond[cond_id] >= 17:
                            exceed = True
                        else:
                            cnt_init_cond[cond_id] += 1
                        break
                if exceed: continue
                if len(traj1) > max_length[cond_id]: 
                    cnt_init_cond[cond_id] -= 1
                    continue
            else:
                if cnt == 151: break
            cnt += 1
            trajs.append([traj1[0:T], traj2[0:T]])
            if i > 1 and i < 6:
                labels.append([-1, 1])
            elif i == 6:
                labels.append([2, 2])
            else:
                labels.append([0, 1])
            # print(trajs)
        print(vid_dir)

    data = prepare_dataset(trajs, 
                           dT=1 if args.source == 0 or args.source == 3 else 5,
                           remove_last=False)#args.source != 0 and args.source != -3)
    pickle.dump(data, open(str(p / 'data.pik'), 'wb'))
    print('Number of videos:', len(trajs))
    print(cnt_init_cond)