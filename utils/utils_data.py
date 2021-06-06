from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
from .utils_vec import *


MASS = math.pi
cx, cy = 16, 12
half_x, half_y, door_length = 8, 8, 3


def phi(x1, x2):
    points = [(-8, 5), (-8, -8), (8, -8), (8, 8), (-8, 8)]
    p1 = (x1[0] - 16, x1[1] - 12)
    p2 = (x2[0] - 16, x2[1] - 12)
    feat = [p1[0], p1[1]]
    feat += [p1[0] - p2[0], p1[1] - p2[1]]
    for point in points:
        feat += [p1[0] - point[0], p1[1] - point[1]]
    feat += [p2[0], p2[1]]
    feat += [p2[0] - p1[0], p2[1] - p1[1]]
    for point in points:
        feat += [p2[0] - point[0], p2[1] - point[1]]
    return feat


def smooth_traj(traj, dT=1):
    smoothed_traj = list(traj)
    T = len(traj)
    for t in range(dT, T - dT, 1):
        pos = (sum([traj[tt][0] for tt in range(t - dT, t + dT + 1)]) / (dT * 2 + 1),
               sum([traj[tt][1] for tt in range(t - dT, t + dT + 1)]) / (dT * 2 + 1))
        smoothed_traj[t] = pos
    return smoothed_traj  


def prepare_physical_data(trajs_raw, max_episode_length=None, dT=1, scale=1.0):
    trajs, vels, accs, Fs, segs = [], [], [], [], []
    for episode in trajs_raw:
        # print(episode[0])
        if max_episode_length is not None:
            max_len = min(dT * max_episode_length, len(episode[0]))
        else:
            max_len = len(episode[0])
        traj1, traj2 = episode[0][:max_len], episode[1][:max_len]
        T = len(traj1)
        # print(T, len(episode[0]))
        vels1, vels2 = [], []
        for t in range(0, T - dT, dT):
            vel1 = rescale((traj1[t + 1][0] - traj1[t][0], traj1[t + 1][1] - traj1[t][1]), dT)
            vel2 = rescale((traj2[t + 1][0] - traj2[t][0], traj2[t + 1][1] - traj2[t][1]), dT)
            vels1.append(vel1)
            vels2.append(vel2)
        trajs.append([traj1[:T-dT:dT], traj2[:T-dT:dT]])
        vels.append([vels1, vels2])
        # print(vels1)
        # input('press any key to continue...')
        T = len(vels1)
        accs1, accs2 = [], []
        for t in range(0, T - 1):
            acc1 = (vels1[t + 1][0] - vels1[t][0], vels1[t + 1][1] - vels1[t][1])
            acc2 = (vels2[t + 1][0] - vels2[t][0], vels2[t + 1][1] - vels2[t][1])
            accs1.append(acc1)
            accs2.append(acc2)
        # print(accs1[:1])
        accs.append([accs1, accs2])
        Fs1 = [rescale(a, MASS) for a in accs1]
        Fs2 = [rescale(a, MASS) for a in accs2]
        Fs.append([Fs1, Fs2])
        # print(len(trajs[-1][0]), len(vels1), len(accs1), len(Fs1))
        segs.append([
            [(cx - half_x * scale, cy + half_y * scale), (cx + half_x * scale, cy + half_y * scale)],
            [(cx + half_x * scale, cy + half_y * scale), (cx + half_x * scale, cy - half_y * scale)],
            [(cx + half_x * scale, cy - half_y * scale), (cx - half_x * scale, cy - half_y * scale)],
            [(cx - half_x * scale, cy - half_y * scale), (cx - half_x * scale, cy + half_y * scale - door_length * scale)]
            ])

    return {'trajs': trajs, 'vels': vels, 'segs': segs, 'accs': accs, 'Fs': Fs}


def prepare_social_data(trajs_raw, max_episode_length=None, dT=1):
    trajs, vels, accs, Fs, segs = [], [], [], [], []
    for episode in trajs_raw:
        # print(episode[0])
        if max_episode_length is not None:
            max_len = min(dT * max_episode_length, len(episode[0]))
        else:
            max_len = len(episode[0])
        traj1, traj2 = smooth_traj(episode[0][:max_len]), smooth_traj(episode[1][:max_len])
        # print(traj1)
        T = len(traj1)
        vels1, vels2 = [], []
        for t in range(0, T - dT, dT):
            vel1 = rescale((traj1[t + 1][0] - traj1[t][0], traj1[t + 1][1] - traj1[t][1]), dT)
            vel2 = rescale((traj2[t + 1][0] - traj2[t][0], traj2[t + 1][1] - traj2[t][1]), dT)
            vels1.append(vel1)
            vels2.append(vel2)
        trajs.append([traj1[0:T-dT:dT], traj2[0:T-dT:dT]])
        vels.append([vels1, vels2])
        # print(vels1)
        # input('press any key to continue...')
        T = len(vels1)
        accs1, accs2 = [], []
        for t in range(0, T - 1):
            acc1 = (vels1[t + 1][0] - vels1[t][0], vels1[t + 1][1] - vels1[t][1])
            acc2 = (vels2[t + 1][0] - vels2[t][0], vels2[t + 1][1] - vels2[t][1])
            accs1.append(acc1)
            accs2.append(acc2)
        # print(accs1[:1])
        accs.append([accs1, accs2])
        Fs1 = [rescale(a, MASS) for a in accs1]
        Fs2 = [rescale(a, MASS) for a in accs2]
        Fs.append([Fs1, Fs2])
        # print(len(trajs[-1][0]), len(vels1), len(accs1), len(Fs1))
        segs.append([
            [(cx - half_x, cy + half_y), (cx + half_x, cy + half_y)],
            [(cx + half_x, cy + half_y), (cx + half_x, cy - half_y)],
            [(cx + half_x, cy - half_y), (cx - half_x, cy - half_y)],
            [(cx - half_x, cy - half_y), (cx - half_x, cy + half_y - door_length)]
            ])

    return {'trajs': trajs, 'vels': vels, 'segs': segs, 'accs': accs, 'Fs': Fs}
