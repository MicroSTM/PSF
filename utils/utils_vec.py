from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import math


def dist(pos1, pos2):
    """distance to another point"""
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    

def rotate(pos, degree):
    x, y = pos[0], pos[1]
    x, y = x * math.cos(math.pi * degree) + y * math.sin(math.pi * degree), -x * math.sin(math.pi * degree) + y * math.cos(math.pi * degree)
    return (x, y)


def cart_to_polar(pos1, pos2):
    d = dist(pos1, pos2)
    theta = math.atan2(pos1[1] - pos2[1], pos1[0] - pos2[0])
    return (d, theta)


def polar_to_cart(norm, dir):
    return (norm * math.cos(dir), norm * math.sin(dir))


def dist_to_seg(pos, seg):
    """distance to a segment"""
    point = np.array(pos)
    line = [np.array(seg[0]), np.array(seg[1])]
    unit_line = line[1] - line[0]
    l2 = np.linalg.norm(unit_line) ** 2
    if l2 < 1e-8:
        projection = line[0]
    else:
        t = max(0, min(1, np.dot(point - line[0], line[1] - line[0]) / l2))
        projection = line[0] + t * (line[1] - line[0])
    n = point - projection
    return np.linalg.norm(n)


def cart_to_polar_seg(pos, seg):
    """polar coord from distance to a line segment"""
    point = np.array(pos)
    line = [np.array(seg[0]), np.array(seg[1])]
    unit_line = line[1] - line[0]
    l2 = np.linalg.norm(unit_line) ** 2
    if l2 < 1e-8:
        projection = line[0]
    else:
        t = max(0, min(1, np.dot(point - line[0], line[1] - line[0]) / l2))
        projection = line[0] + t * (line[1] - line[0])
    n = point - projection
    return (np.linalg.norm(n), math.atan2(n[1], n[0]))


def proj_dir(vec, dir):
    """coord projected to a direction"""
    return math.cos(dir - math.atan2(vec[1], vec[0])) * norm(vec)


def cart_to_polar_proj(pos, dir):
    """polar coord from projected to a given direction"""
    return (proj_dir(pos, dir), dir)


def get_vec(pos1, pos2, scale=None):
    """pos1 -> pos2"""
    s = scale if scale is not None else 1.0
    return ((pos2[0] - pos1[0]) * s, (pos2[1] - pos1[1]) * s)


def norm(vec):
    return (vec[0] ** 2 + vec[1] ** 2) ** 0.5


def rescale(vec, scale):
    return (vec[0] * scale, vec[1] * scale)


def vec_sum(vec1, vec2):
    return (vec1[0] + vec2[0], vec1[1] + vec2[1])


def get_ave_vel(trajs):
    ave_vel_list = []
    for traj in trajs:
        ave_vel = 0
        for t in range(1, len(traj)):
            cur_vel = dist(traj[t], traj[t - 1])
            ave_vel += cur_vel
        if len(traj) > 1:
            ave_vel /= len(traj) - 1
        ave_vel_list.append(ave_vel)
    return ave_vel_list


def get_vel(trajs):
    """"""
    vels = [None] * len(trajs)
    for traj_id, traj in enumerate(trajs):
        length = len(traj)
        vels[traj_id] = [0] * (length - 1)
        for t in range(1, length):
            cur_vel = get_vec(traj[t - 1], traj[t])
            vels[traj_id][t - 1] = cur_vel
    return vels


def get_accs(vels, dT):
    """get accelerations from velocities"""
    T = len(vels)
    accs = [None] * (T - 1)
    for t in range(T - 1):
        accs[t] = get_vec(vels[t], vels[t + 1], 1.0 / dT)
    return accs


def get_Fs(accs, mass):
    """get forces from accelerations and mass"""
    # print(accs)
    return [rescale(a, mass) for a in accs]


def x_v_array(pos_list, vel_list):
    """convert the positions and velocities of all entities into a numpy array"""
    N = len(pos_list)
    arr = np.zeros((N, 4))
    for i in range(N):
        pos, vel = pos_list[i], vel_list[i]
        arr[i, :] = np.array([pos[0], pos[1], vel[0], vel[1]])
    return arr


def vec_seq_to_array(vecs):
    return np.vstack([np.array(vec) for vec in vecs])


def get_XY(source, qs, forces, N):
    """get input, output from gen coords and forces"""
    X, Y, theta = [], [], []
    T = forces.shape[1]
    if source[0] < N:
        entity_id = source[0]
        for t in range(T):
            q, f = qs[t], forces[entity_id, t, :]
            x, y = q[0], -np.linalg.norm(f) * math.cos(math.atan2(f[1], f[0]) - q[1])
            # if abs(y) > 1e-8:
            #     X.append(np.array(x))
            #     Y.append(np.array(y))
            # print(y / (x - 6))
            if abs(x) < 1e-8:
                y = 0
            X.append(np.array(x))
            Y.append(np.array(y))
            theta.append(np.array(q[1]))

    if source[1] < N:
        entity_id = source[1]
        for t in range(T):
            q, f = qs[t], forces[entity_id, t, :]
            x, y = q[0], np.linalg.norm(f) * math.cos(math.atan2(f[1], f[0]) - q[1])
            # if abs(y) > 1e-8:
            #     X.append(np.array(x))
            #     Y.append(np.array(y))
            # print(y / (x - 6))
            X.append(np.array(x))
            Y.append(np.array(y))
            theta.append(np.array(q[1]))

    if len(X) == 0:
        return None, None, None
    return np.vstack(X), np.vstack(Y), np.vstack(theta)


def get_feat(source, qs, N, remove_last=True):
    """get input from gen coords"""
    feat_X = {entity_id: [] for entity_id in range(N)}
    feat_Y = {entity_id: [] for entity_id in range(N)}
    T = len(qs)
    if remove_last:
        T -= 1
    if source[0] < N:
        entity_id = source[0]
        for t in range(T):
            q = qs[t]
            x, theta = q[0], q[1]
            feat_X[entity_id].append(np.array([-x * math.cos(theta), -math.cos(theta)]))
            feat_Y[entity_id].append(np.array([-x * math.sin(theta), -math.sin(theta)])) 

    if source[1] < N:
        entity_id = source[1]
        for t in range(T):
            q = qs[t]
            x, theta = q[0], q[1] + math.pi
            feat_X[entity_id].append(np.array([-x * math.cos(theta), -math.cos(theta)]))
            feat_Y[entity_id].append(np.array([-x * math.sin(theta), -math.sin(theta)])) 
    for entity_id in range(N):
        if not feat_X[entity_id]:
            feat_X[entity_id] = [np.array([0, 0])] * T
        if not feat_Y[entity_id]:
            feat_Y[entity_id] = [np.array([0, 0])] * T

    if len(feat_X[0]) == 0:
        return None, None
    return {entity_id: np.vstack(feat_X[entity_id]) for entity_id in range(N)}, \
           {entity_id: np.vstack(feat_Y[entity_id]) for entity_id in range(N)}


def get_feat_potentials(source, qs, N, remove_last=True):
    """get input from gen coords (for potential functions)"""
    feat = {entity_id: [] for entity_id in range(N)}
    T = len(qs)
    if remove_last:
        T -= 1
    if source[0] < N:
        entity_id = source[0]
        for t in range(T):
            q = qs[t]
            x, theta = q[0], q[1]
            feat[entity_id].append(np.array([0.5 * x ** 2, x]))

    if source[1] < N:
        entity_id = source[1]
        for t in range(T):
            q = qs[t]
            x, theta = q[0], q[1] + math.pi
            feat[entity_id].append(np.array([0.5 * x ** 2, x]))
    for entity_id in range(N):
        if not feat[entity_id]:
            feat[entity_id] = [np.array([0, 0])] * T

    if len(feat[0]) == 0:
        return None, None
    return {entity_id: np.vstack(feat[entity_id]) for entity_id in range(N)}


def _get_id(x, min_x, max_x, dx):
    N_x = int((max_x - min_x) / dx + 1)
    # if x < min_x: return 0
    # if x > max_x: return (max_x - min_x) // dx
    return max(min(int((x - min_x) / dx + 0.5), N_x - 1), 0)


# def get_id(pos1, pos2, min_x, max_x, dx, min_y, max_y, dy):
#     """get region id"""
#     min_x, max_y = 16 - 0, 12 - 0
#     if pos1[0] < min_x: 
#         index = 0 if pos1[1] < max_y else 1
#     else:
#         index = 2 if pos1[1] >= max_y else 3
#     if pos1[0] < min_x and pos1[1] > max_y:
#         index = 0
#     else:
#         index = 1
#     id_vec = np.zeros(2)
#     id_vec[index] = 1
#     # print(pos1, index)
#     return id_vec


def in_bound(pos, bound):
    return pos[0] >= bound[0] and pos[0] <= bound[1] \
       and pos[1] >= bound[2] and pos[1] <= bound[3]


def get_id(pos1, pos2, min_x, max_x, dx, min_y, max_y, dy, bound=[16-7, 16+7, 12-7, 12+7]):
    """get region id"""
    N_x = int((max_x - min_x) / dx + 1)
    N_y = int((max_y - min_y) / dy + 1)
    dim = N_x * N_y
    dim_all = dim
    id_vec = np.zeros(dim_all)
    # if not in_bound(pos1, bound):
    #     return id_vec

    id1 = int(_get_id(pos1[0], min_x, max_x, dx) * N_y + _get_id(pos1[1], min_y, max_y, dy))
    if id1 < dim_all: # if out size of the range, then return all zero vector
        id_vec[id1] = 1
    return id_vec


def get_feat_0(source, qs, N, remove_last=True):
    """get input from gen coords"""
    feat_X = {entity_id: [] for entity_id in range(N)}
    feat_Y = {entity_id: [] for entity_id in range(N)}
    T = len(qs)
    if remove_last:
        T -= 1
    if source[0] < N:
        entity_id = source[0]
        for t in range(T):
            q = qs[t]
            x, theta = q[0], q[1]
            feat_X[entity_id].append(np.array([-math.cos(theta)]))
            feat_Y[entity_id].append(np.array([-math.sin(theta)])) 

    if source[1] < N:
        entity_id = source[1]
        for t in range(T):
            q = qs[t]
            x, theta = q[0], q[1] + math.pi
            feat_X[entity_id].append(np.array([-math.cos(theta)]))
            feat_Y[entity_id].append(np.array([-math.sin(theta)])) 
    for entity_id in range(N):
        if not feat_X[entity_id]:
            feat_X[entity_id] = [np.array([0])] * T
        if not feat_Y[entity_id]:
            feat_Y[entity_id] = [np.array([0])] * T

    if len(feat_X[0]) == 0:
        return None, None
    return {entity_id: np.vstack(feat_X[entity_id]) for entity_id in range(N)}, \
           {entity_id: np.vstack(feat_Y[entity_id]) for entity_id in range(N)}


def get_feat_nobias(source, qs, N, remove_last=True):
    """get input from gen coords"""
    feat_X = {entity_id: [] for entity_id in range(N)}
    feat_Y = {entity_id: [] for entity_id in range(N)}
    T = len(qs)
    if remove_last:
        T -= 1
    if source[0] < N:
        entity_id = source[0]
        for t in range(T):
            q = qs[t]
            x, theta = q[0], q[1]
            feat_X[entity_id].append(np.array([-x * math.cos(theta)]))
            feat_Y[entity_id].append(np.array([-x * math.sin(theta)])) 

    if source[1] < N:
        entity_id = source[1]
        for t in range(T):
            q = qs[t]
            x, theta = q[0], q[1] + math.pi
            feat_X[entity_id].append(np.array([-x * math.cos(theta)]))
            feat_Y[entity_id].append(np.array([-x * math.sin(theta)])) 
    for entity_id in range(N):
        if not feat_X[entity_id]:
            feat_X[entity_id] = [np.array([0])] * T
        if not feat_Y[entity_id]:
            feat_Y[entity_id] = [np.array([0])] * T

    if len(feat_X[0]) == 0:
        return None, None
    return {entity_id: np.vstack(feat_X[entity_id]) for entity_id in range(N)}, \
           {entity_id: np.vstack(feat_Y[entity_id]) for entity_id in range(N)}


def get_feat0(source, qs, N, remove_last=True):
    """get input from gen coords"""
    feat_X = {entity_id: [] for entity_id in range(N)}
    feat_Y = {entity_id: [] for entity_id in range(N)}
    T = len(qs)
    if remove_last:
        T -= 1
    if source[0] < N and source[1] < N:
        entity_id = source[0]
        for t in range(T):
            q = qs[t]
            x, theta = q[0], q[1]
            feat_X[entity_id].append(np.array([-x * math.cos(theta), -math.cos(theta)]))
            feat_Y[entity_id].append(np.array([-x * math.sin(theta), -math.sin(theta)])) 

        entity_id = source[1]
        for t in range(T):
            q = qs[t]
            x, theta = q[0], q[1] + math.pi
            feat_X[entity_id].append(np.array([-x * math.cos(theta), -math.cos(theta)]))
            feat_Y[entity_id].append(np.array([-x * math.sin(theta), -math.sin(theta)])) 
    else:
        entity_id = source[0]
        for t in range(T):
            q = qs[t]
            x, theta = q[0], q[1]
            feat_X[entity_id].append(np.array([0, -math.cos(theta)]))
            feat_Y[entity_id].append(np.array([0, -math.sin(theta)])) 

    for entity_id in range(N):
        if not feat_X[entity_id]:
            feat_X[entity_id] = [np.array([0, 0])] * T
        if not feat_Y[entity_id]:
            feat_Y[entity_id] = [np.array([0, 0])] * T

    if len(feat_X[0]) == 0:
        return None, None
    return {entity_id: np.vstack(feat_X[entity_id]) for entity_id in range(N)}, \
           {entity_id: np.vstack(feat_Y[entity_id]) for entity_id in range(N)}


def get_feat_1(source, qs, N, remove_last=True):
    """get input from gen coords"""
    feat_X = {entity_id: [] for entity_id in range(N)}
    feat_Y = {entity_id: [] for entity_id in range(N)}
    T = len(qs)
    if remove_last:
        T -= 1
    if source[0] < N:
        entity_id = source[0]
        for t in range(T):
            q = qs[t]
            x, theta = q[0], q[1]
            feat_X[entity_id].append(np.array([-1 / (x**2) * math.cos(theta)]))
            feat_Y[entity_id].append(np.array([-1 / (x**2) * math.sin(theta)])) 

    if source[1] < N:
        entity_id = source[1]
        for t in range(T):
            q = qs[t]
            x, theta = q[0], q[1] + math.pi
            feat_X[entity_id].append(np.array([-1 / (x**2) * math.cos(theta)]))
            feat_Y[entity_id].append(np.array([-1 / (x**2) * math.sin(theta)])) 
    for entity_id in range(N):
        if not feat_X[entity_id]:
            feat_X[entity_id] = [np.array([0])] * T
        if not feat_Y[entity_id]:
            feat_Y[entity_id] = [np.array([0])] * T

    if len(feat_X[0]) == 0:
        return None, None
    return {entity_id: np.vstack(feat_X[entity_id]) for entity_id in range(N)}, \
           {entity_id: np.vstack(feat_Y[entity_id]) for entity_id in range(N)}


def get_feat_3(source, qs, N, remove_last=True):
    """get input from gen coords"""
    feat_X = {entity_id: [] for entity_id in range(N)}
    feat_Y = {entity_id: [] for entity_id in range(N)}
    T = len(qs)
    if remove_last:
        T -= 1
    if source[0] < N:
        entity_id = source[0]
        for t in range(T):
            q = qs[t]
            x, theta = q[0], q[1]
            square_term = 1 / max(x ** 2, 1e-6) #if abs(x) > 1e-6 else 0
            feat_X[entity_id].append(np.array([square_term * math.cos(theta), -x * math.cos(theta), -math.cos(theta)]))
            feat_Y[entity_id].append(np.array([square_term * math.sin(theta), -x * math.sin(theta), -math.sin(theta)])) 

    if source[1] < N:
        entity_id = source[1]
        for t in range(T):
            q = qs[t]
            x, theta = q[0], q[1] + math.pi
            square_term = 1 / max(x ** 2, 1e-6) #if abs(x) > 1e-6 else 0
            feat_X[entity_id].append(np.array([square_term * math.cos(theta), -x * math.cos(theta), -math.cos(theta)]))
            feat_Y[entity_id].append(np.array([square_term * math.sin(theta), -x * math.sin(theta), -math.sin(theta)])) 
    for entity_id in range(N):
        if not feat_X[entity_id]:
            feat_X[entity_id] = [np.array([0, 0, 0])] * T
        if not feat_Y[entity_id]:
            feat_Y[entity_id] = [np.array([0, 0, 0])] * T

    if len(feat_X[0]) == 0:
        return None, None
    return {entity_id: np.vstack(feat_X[entity_id]) for entity_id in range(N)}, \
           {entity_id: np.vstack(feat_Y[entity_id]) for entity_id in range(N)}


def decomp_vec(vec, dir):
    """decompose a vec into two components, one of which is along a given direction dir"""
    proj_norm = proj_dir(vec, dir)
    vec1 = (proj_norm * math.cos(dir), proj_norm * math.sin(dir))
    vec2 = (vec[0] - vec1[0], vec[1] - vec1[1])
    return vec1, vec2


def rotate_dir(dir, rotate_dir):
    """rotate"""
    new_dir = dir + rotate_dir
    if new_dir > math.pi:
        new_dir -= math.pi * 2
    elif new_dir < -math.pi:
        new_dir += math.pi * 2
    return new_dir