from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from utils import *


np.set_printoptions(precision=3, suppress=True)
RADIUS = 1.0
MASS   = math.pi * (RADIUS ** 2)
cx, cy = 16, 12 # center
half_x, half_y = 8, 8
door_length = 3
points = [(8, 20), (8, 20 - 3), (8, 4), (24, 4), (24, 20)] # reference points


class GenCoordModel_Social_All_Coord():
    """
    Baseline: Using all generalized coordinates
    """
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.x_range = (cx - half_x, cx + half_x)
        self.y_range = (cy - half_y, cy + half_y)
        self.dx, self.dy = 8, 8
        self.min_x, self.max_x = 16 - 8 + self.dx // 2, 16 + 8 - self.dx // 2
        self.min_y, self.max_y = 12 - 8 + self.dy // 2, 16 + 8 - self.dy // 2


    def set_seed(self, seed):
        np.random.seed(seed)


    def gen_coord_pool(self, trajs, vels, segs, Fs=None):
        """pairs of entities, entities and segments"""
        N = len(trajs)
        M = len(segs)
        sources = []
        types = []
        gen_coords_all = []

        # Type I: 
        # other entities
        for entity_id_1 in range(0, N - 1):
            for entity_id_2 in range(entity_id_1 + 1, N):
                sources.append((entity_id_1, entity_id_2))
                types.append('entity')
                T = len(trajs[entity_id_1])
                coords = []
                for t in range(T):
                    coord = cart_to_polar(trajs[entity_id_1][t], trajs[entity_id_2][t])
                    coords.append(coord)
                gen_coords_all.append(coords)
        # reference points
        for entity_id, traj in enumerate(trajs):
            for point_id, point in enumerate(points):
                sources.append((entity_id, N + point_id))
                types.append('ref_point')
                T = len(traj)
                coords = []
                for t in range(T):
                    coord = cart_to_polar(traj[t], point)
                    if t < T - 1 and Fs is not None and entity_id == 0 and point_id == 0:
                        coord_after = cart_to_polar(traj[t + 1], point)
                    coords.append(coord)
                gen_coords_all.append(coords)
        
        # Type II: for constraint violation
        # other entities
        for entity_id_1 in range(0, N - 1):
            for entity_id_2 in range(entity_id_1 + 1, N):
                sources.append((entity_id_1, entity_id_2))
                types.append('violation_entity')
                T = len(trajs[entity_id_1])
                coords = []
                for t in range(T):
                    coord = cart_to_polar(trajs[entity_id_1][t], trajs[entity_id_2][t])
                    tmp_pos1 = vec_sum(trajs[entity_id_1][t], vels[entity_id_1][t])
                    tmp_pos2 = vec_sum(trajs[entity_id_2][t], vels[entity_id_2][t])
                    after_dist = dist(tmp_pos1, tmp_pos2)
                    if after_dist < RADIUS * 2 - 1e-8:
                        last = True
                        if last:
                            coord = (coord[0] - after_dist, coord[1])
                            coords.append(coord)
                        else:
                            coords.append((0, coord[1]))
                    else:
                        coords.append((0, coord[1]))
                gen_coords_all.append(coords)
        # segments
        for entity_id, (traj, vel) in enumerate(zip(trajs, vels)):
            for seg_id, seg in enumerate(segs):
                sources.append((entity_id, N + seg_id))
                types.append('violation_segments')
                T = len(traj)
                coords = []
                for t in range(T):
                    coord = cart_to_polar_seg(traj[t], seg)
                    tmp_pos = vec_sum(traj[t], vel[t])
                    after_dist = dist_to_seg(tmp_pos, seg)
                    if after_dist < RADIUS - 1e-8:
                        coord = (coord[0] - after_dist, coord[1])
                        if t < T - 1:
                            acc = cart_to_polar(vels[entity_id][t + 1], vels[entity_id][t])
                        coords.append(coord)
                    else:
                        coords.append((0, coord[1]))
                gen_coords_all.append(coords)

        return {'sources': sources, 'types': types, 
                'gen_coords': gen_coords_all, 
                'num_entities': N, 'num_segs': M}


    def prepare_data(self, gen_coords, Fs=None, selected=None, entities=None, remove_last=True):
        """prepare feature vectors for forces"""
        feat_X, feat_Y  = {}, {}
        if Fs:
            gt_X, gt_Y = [], []
            for gen_coords_one_episode, forces in zip(gen_coords, Fs):
                sources = gen_coords_one_episode['sources']
                gen_coords_all = gen_coords_one_episode['gen_coords']
                N = gen_coords_one_episode['num_entities']
                if entities is None:
                    entities = list(range(N))
                for gen_coord_id, (source, qs) in enumerate(zip(sources, gen_coords_all)):
                    if selected and gen_coord_id not in selected: continue
                    cur_feat_X, cur_feat_Y = get_feat(source, qs, N, remove_last=remove_last)
                    if gen_coord_id not in feat_X:
                        feat_X[gen_coord_id], feat_Y[gen_coord_id] = [], []
                    for entity_id in entities:
                        feat_X[gen_coord_id].append(cur_feat_X[entity_id])
                        feat_Y[gen_coord_id].append(cur_feat_Y[entity_id])
                for entity_id in entities:
                    gt_X.append(np.vstack([np.array(f[0]) for f in forces[entity_id]])) 
                    gt_Y.append(np.vstack([np.array(f[1]) for f in forces[entity_id]]))
            return feat_X, feat_Y, gt_X, gt_Y
        else:
            for gen_coords_one_episode in gen_coords:
                sources = gen_coords_one_episode['sources']
                gen_coords_all = gen_coords_one_episode['gen_coords']
                N = gen_coords_one_episode['num_entities']
                if entities is None:
                    entities = list(range(N))
                for gen_coord_id, (source, qs) in enumerate(zip(sources, gen_coords_all)):
                    cur_feat_X, cur_feat_Y = get_feat(source, qs, N, remove_last=remove_last)
                    if gen_coord_id not in feat_X:
                        feat_X[gen_coord_id], feat_Y[gen_coord_id] = [], []
                    for entity_id in entities:
                        feat_X[gen_coord_id].append(cur_feat_X[entity_id])
                        feat_Y[gen_coord_id].append(cur_feat_Y[entity_id])
            return feat_X, feat_Y


    def prepare_data_potentials(self, gen_coords, Fs=None, selected=None, entities=None, remove_last=True):
        """prepare feature vectors for potentials"""
        feat  = {}
        if Fs:
            gt = []
            for gen_coords_one_episode, forces in zip(gen_coords, Fs):
                sources = gen_coords_one_episode['sources']
                gen_coords_all = gen_coords_one_episode['gen_coords']
                N = gen_coords_one_episode['num_entities']
                if entities is None:
                    entities = list(range(N))
                for gen_coord_id, (source, qs) in enumerate(zip(sources, gen_coords_all)):
                    if selected and gen_coord_id not in selected: continue
                    cur_feat = get_feat_potentials(source, qs, N, remove_last=remove_last)
                    if gen_coord_id not in feat:
                        feat[gen_coord_id]= []
                    for entity_id in entities:
                        feat[gen_coord_id].append(cur_feat[entity_id])
                for entity_id in entities:
                    gt.append(np.vstack([np.array(p) for p in potentials[entity_id]])) 
            return feat, gt
        else:
            for gen_coords_one_episode in gen_coords:
                sources = gen_coords_one_episode['sources']
                gen_coords_all = gen_coords_one_episode['gen_coords']
                N = gen_coords_one_episode['num_entities']
                if entities is None:
                    entities = list(range(N))
                for gen_coord_id, (source, qs) in enumerate(zip(sources, gen_coords_all)):
                    cur_feat = get_feat_potentials(source, qs, N, remove_last=remove_last)
                    if gen_coord_id not in feat:
                        feat[gen_coord_id] = []
                    for entity_id in entities:
                        feat[gen_coord_id].append(cur_feat[entity_id])
            return feat


    def pred_forces(self, reg, mask, gen_coords, selected, entities, remove_last=True):
        """pred forces given selected generalized coordinates and the fitted model"""
        feat_X, feat_Y = self.prepare_data([gen_coords], entities=entities, remove_last=remove_last)
        feat_X = np.concatenate([np.vstack(feat_X[sel_id]) for sel_id in selected], axis=-1)
        feat_Y = np.concatenate([np.vstack(feat_Y[sel_id]) for sel_id in selected], axis=-1)
        forces = np.concatenate([reg.predict(feat_X[:,mask].reshape(feat_X.shape[0], -1)).reshape(-1, 1), 
                                 reg.predict(feat_Y[:,mask].reshape(feat_Y.shape[0], -1)).reshape(-1, 1)], axis=-1)
        return forces.reshape(len(entities), -1, 2)


    def pred_potentials(self, reg, mask, gen_coords, selected, entities, remove_last=True):
        """pred forces given selected generalized coordinates and the fitted model"""
        feat= self.prepare_data_potentials([gen_coords], entities=entities, remove_last=remove_last)
        feat = np.concatenate([np.vstack(feat[sel_id]) for sel_id in selected], axis=-1)
        potentials = reg.predict(feat[:,mask].reshape(feat.shape[0], -1)).reshape(-1, 1)
        return potentials.reshape(len(entities), -1)


    def pred_vels(self, reg, mask, gen_coords, selected, vels, entities, remove_last=True):
        """pred velocities at next time step given selected generalized coordinates and the fitted model"""
        forces = self.pred_forces(reg, mask, gen_coords, selected, entities, remove_last=remove_last)
        accs = forces / MASS
        pred = np.array([vel for entity_id, vel in enumerate(vels) if entity_id in entities])
        pred[:, 1:, :] = pred[:, :-1, :] + accs
        return pred[:, 1:, ]


    def get_region_indices(self, trajs, entity_id, remove_last=True):
        """get the indices of descritized regions for a target entity_id"""
        T = len(trajs[entity_id])
        if remove_last: 
            T -= 1
        region_indices = [None] * T
        for t in range(T):
            region_indices[t] = get_id(trajs[entity_id][t], trajs[1 - entity_id][t], 
                                       self.min_x, self.max_x, self.dx,
                                       self.min_y, self.max_y, self.dy)
        return region_indices


    def log_likelihood(self, log_Z, Lambda, regions, responses, remove_last=True):
        """get log likelihood"""
        return (regions * Lambda * responses).sum(1).mean() - log_Z


    def train(self, data, train_size, entities=[0,1]):
        trajs, vels, Fs, segs = data['trajs'], data['vels'], data['Fs'], data['segs']
        num_episodes = min(len(trajs), train_size)
        trajs, vels, Fs, segs = trajs[:train_size], vels[:train_size], Fs[:train_size], segs[:train_size]
        f_gt_list_full = [np.stack([np.vstack([np.array(f) for f in forces]) \
                                        for forces in Fs[episode_id]]) \
                            for episode_id in range(num_episodes)]
        region_list = [np.stack([np.vstack([self.get_region_indices(trajs[episode_id], entity_id)]) \
                                        for entity_id in entities]) \
                            for episode_id in range(num_episodes)]
        f_gt_list = [np.stack([np.vstack([np.array(f) for f in Fs[episode_id][entity_id]]) \
                                        for entity_id in entities]) \
                            for episode_id in range(num_episodes)]
        f_gt = np.concatenate([f.reshape(-1, 2) for f in f_gt_list], 0)
        vel_gt_list = [np.stack([np.vstack([np.array(v) for v in vels[episode_id][entity_id]]) \
                                            for entity_id in entities]) \
                            for episode_id in range(num_episodes)]
        vel_gt_list_noinit = [np.stack([np.vstack([np.array(v) for v in vels[episode_id][entity_id][1:]]) \
                                            for entity_id in entities]) \
                            for episode_id in range(num_episodes)]
        vel_gt = np.concatenate([v.reshape(-1, 2) for v in vel_gt_list_noinit], 0)
        gen_coords = [self.gen_coord_pool(trajs[episode_id], vels[episode_id], segs[episode_id], Fs=Fs[episode_id]) \
                            for episode_id in range(num_episodes)]
        feat_X, feat_Y, gt_X, gt_Y = self.prepare_data(gen_coords, f_gt_list_full, entities=entities)
        gt = np.vstack(gt_X + gt_Y)
        sources = gen_coords[0]['sources']
        types = gen_coords[0]['types']

        gen_coord_list = range(len(sources))
        selected = list(gen_coord_list)
        selected_types = list(types)
        final_reg = None
        feat = np.concatenate([np.vstack(feat_X[sel_id] + feat_Y[sel_id]) \
                            for sel_id in selected], axis=-1)

        num_samples = feat.shape[0]
        N_train = num_samples
        X_train, Y_train = feat[:N_train, :], gt[:N_train, :].reshape(-1)

        alpha = 0.1
        reg = Lasso(alpha=alpha, fit_intercept=False).fit(X_train, Y_train)
        Y_pred = reg.predict(X_train)
        err = Y_pred - Y_train
        mean_err = err.mean()
        std_err = err.std()

        X_train = X_train[abs(err - mean_err) < std_err * 2]
        Y_train = Y_train[abs(err - mean_err) < std_err * 2]
        reg = Lasso(alpha=alpha, fit_intercept=False).fit(X_train, Y_train)
        mask = np.abs(reg.coef_) > 1e-2
        cnt = X_train.shape[0]
        X_train = X_train[:, mask].reshape(cnt, -1)
        reg = Ridge(alpha=0.01, fit_intercept=False).fit(X_train, Y_train)

        # predict
        f_pred_list = [self.pred_forces(reg, mask, gen_coords[episode_id], selected, entities=entities)
                        for episode_id in range(num_episodes)]
        f_pred = np.concatenate([f.reshape(-1, 2) for f in f_pred_list], 0)
        loss = mean_squared_error(f_gt.reshape(-1, 2), f_pred.reshape(-1, 2))

        final_reg = reg
        final_mask = mask
        f_pred_list = [self.pred_forces(final_reg, final_mask, gen_coords[episode_id], selected, entities=entities)
                                for episode_id in range(num_episodes)]
        all_cos = [None] * num_episodes
        appear = [None] * num_episodes
        for episode_id, (vel_gt, f_pred, region) in enumerate(zip(vel_gt_list_noinit, f_pred_list, region_list)):
            A, B = vel_gt.reshape(-1, 2).copy(), f_pred.reshape(-1, 2).copy()
            for i in range(A.shape[0]):
                if np.linalg.norm(A[i,:]) < 0.3 and np.linalg.norm(B[i,:]) < 0.2:
                    A[i,:] = [1, 1]
                    B[i,:] = [1, 1]            
            cos_AB = np.nan_to_num(((A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))))
            all_cos[episode_id] = (region[0] * cos_AB.reshape(-1,1)).mean(0).reshape(1,-1)
            appear[episode_id] = np.clip(region[0].sum(0), 0, 1).reshape(1,-1)
            # print((region[0] * cos_AB.reshape(-1,1)).mean(0))
        # print(np.concatenate(all_cos, axis=0).sum(0).shape,
        #       np.concatenate(region_list, axis=1)[0].sum(0).shape)
        # Lambda = np.concatenate(all_cos, axis=0).mean(0)
        Lambda = np.concatenate(all_cos, axis=0).sum(0) / np.concatenate(appear, axis=0).sum(0)
        # Lambda = np.concatenate(all_cos, axis=0).sum(0) / np.concatenate(region_list, axis=1)[0].sum(0)
        Lambda = np.nan_to_num(Lambda)
        # if np.abs(Lambda.min()) < 1e-8:
        #     Lambda[np.abs(Lambda) < 1e-8] = 0.66
        #     Lambda /= Lambda.sum()
        # Lambda /= Lambda.sum()
        # log_Z = (Lambda * np.concatenate(all_cos, axis=0).mean(0)).sum()
        log_Z = 0 # expected filter responses for background motion is 0

        # N_x = int((self.max_x - self.min_x) / self.dx + 1)
        # N_y = int((self.max_y - self.min_y) / self.dy + 1)
        # for i_x_1 in range(N_x):
        #     for i_y_1 in range(N_y):
        #         # for i_x_2 in range(N_x):
        #         #     for i_y_2 in range(N_y):
        #         x_1, y_1 = i_x_1 * self.dx + self.min_x, i_y_1 * self.dy + self.min_y
        #         # x_2, y_2 = i_x_2 * self.dx + self.min_x, i_y_2 * self.dy + self.min_y
        #         region_id = (i_x_1 * N_y + i_y_1) #* (N_x * N_y) + (i_x_2 * N_y + i_y_2)
        #         print(region_id, (x_1, y_1), Lambda[region_id])
        print(Lambda)

        if self.model_path:
            pickle.dump({'selected': selected, 'types': selected_types, 
                         'reg': final_reg, 'mask': final_mask,
                         'Lambda': Lambda, 'log_Z': log_Z},
                        open(self.model_path, 'wb'))
        else:
            self.res = {'selected': selected, 'types': selected_types, 
                         'reg': final_reg, 'mask': final_mask,
                         'Lambda': Lambda, 'log_Z': log_Z}


    def test(self, data, entities=[0,1]):
        """test the model"""
        # print(self.model_path)
        model = pickle.load(open(self.model_path, 'rb'))
        selected, reg, mask = model['selected'], model['reg'], model['mask']
        Lambda, log_Z = model['Lambda'], model['log_Z']
        # print(selected, reg.coef_)

        trajs, vels, Fs, segs = data['trajs'], data['vels'], data['Fs'], data['segs']
        num_episodes = len(trajs)
        region_list = [np.stack([np.vstack([self.get_region_indices(trajs[episode_id], entity_id)]) \
                                        for entity_id in entities]) \
                            for episode_id in range(num_episodes)]
        f_gt_list = [np.stack([np.vstack([np.array(f) for f in Fs[episode_id][entity_id]]) \
                                        for entity_id in entities]) \
                            for episode_id in range(num_episodes)]
        f_gt = np.concatenate([f.reshape(-1, 2) for f in f_gt_list], 0)
        vel_gt_list = [np.stack([np.vstack([np.array(v) for v in vels[episode_id][entity_id]]) \
                                            for entity_id in entities]) \
                            for episode_id in range(num_episodes)]
        vel_gt_list_noinit = [np.stack([np.vstack([np.array(v) for v in vels[episode_id][entity_id][1:]]) \
                                            for entity_id in entities]) \
                            for episode_id in range(num_episodes)]
        vel_gt = np.concatenate([v.reshape(-1, 2) for v in vel_gt_list_noinit], 0)
        gen_coords = [self.gen_coord_pool(trajs[episode_id], vels[episode_id], segs[episode_id]) \
                            for episode_id in range(num_episodes)]

        """TODO: different sources in different episodes"""
        sources = gen_coords[0]['sources']

        f_pred_list = [self.pred_forces(reg, mask, gen_coords[episode_id], selected, entities=entities)
                                for episode_id in range(num_episodes)]
        f_pred = np.concatenate([f.reshape(-1, 2) for f in f_pred_list], 0)

        # # print(np.concatenate([f_gt, f_pred], axis=1))

        # # loss = mean_squared_error(f_gt.reshape(-1, 2), f_pred.reshape(-1, 2))
        # loss = mean_squared_error(f_gt.reshape(-1, 2), f_pred.reshape(-1, 2))
        # # print('loss (forces):', loss)
        # # print('average f_gt:', np.linalg.norm(f_gt, axis=1).mean())
        # print('loss (forces, norm):', loss / np.linalg.norm(f_gt, axis=1).mean())
        
        vel_pred_list = [self.pred_vels(reg, mask, gen_coords[episode_id], selected, vels[episode_id], entities=entities)
                                for episode_id in range(num_episodes)]
        vel_pred = np.concatenate([v.reshape(-1, 2) for v in vel_pred_list], 0)
        # loss = mean_squared_error(vel_gt.reshape(-1, 2), vel_pred.reshape(-1, 2))
        # print('loss (velocities):', loss)

        # A, B = vel_gt.reshape(-1, 2).copy(), f_pred.reshape(-1, 2).copy()
        # # A, B = vel_gt.reshape(-1, 2).copy(), vel_pred.reshape(-1, 2).copy()
        # # print(np.concatenate([A, B], axis=1))
        # # print(B)
        # # cos_AB_list = []
        # for i in range(A.shape[0]):
        #     # if np.linalg.norm(A[i,:]) > 0.2:
        #     #     cos_AB_list.append(A[i,:] * B[i,:] / (np.linalg.norm(A[i,:]) * np.linalg.norm(B[i,:])))
        #     if np.linalg.norm(A[i,:]) < 0.2 and np.linalg.norm(B[i,:]) < 0.2 * MASS:
        #     # if np.linalg.norm(B[i:,]) < 0.2:
        #         A[i,:] = [1, 1]
        #         B[i,:] = [1, 1]
        # cos_AB = ((A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)))
        # # cos_AB = cos_AB_list
        # print(cos_AB)
        # # print('cos (velocities):', np.nan_to_num(cos_AB).mean())
        # print('cos (forces):', np.nan_to_num(cos_AB).mean())

        cos_AB_list = []
        # for vel_gt, f_pred in zip(vel_gt_list_noinit[95//2:95//2+1], f_pred_list[95//2:95//2+1]):
        cnt = -1
        for episode_id, (vel_gt, f_pred, region) in enumerate(zip(vel_gt_list_noinit, f_pred_list, region_list)):
        # for vel_gt, f_pred in zip(f_gt_list, f_pred_list):
        # for vel_gt, f_pred in zip(vel_gt_list_noinit, vel_pred_list):
            cnt += 2
            A, B = vel_gt.reshape(-1, 2).copy(), f_pred.reshape(-1, 2).copy()
            for i in range(A.shape[0]):
                if np.linalg.norm(A[i,:]) < 0.3 and np.linalg.norm(B[i,:]) < 0.2:
                    A[i,:] = [1, 1]
                    B[i,:] = [1, 1]
                # elif np.linalg.norm(A[i,:]) < 0.2:
                #     A[i,:] = [0, 0]
                # elif np.linalg.norm(B[i,:]) < 0.2:
                #     B[i,:] = [0, 0]
            
            cos_AB = ((A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)))
            # print('cos (vel-forces):', np.nan_to_num(cos_AB).mean())
            # if cnt == 73:
            #     print(np.concatenate([A, B, cos_AB.reshape(-1,1)], axis=1))
            ll = (region[0] * np.nan_to_num(cos_AB).reshape(-1,1) * Lambda).sum(-1) - log_Z
            # print(ll.mean())
            # cos_AB_list.append(np.nan_to_num(cos_AB))
            cos_AB_list.append(ll)
            # input('press any key to continue...')

        return cos_AB_list


    def visualize_force(self, dx=1.5, dy=1.5, entity_id=0):
        """visualize the force fields for an entitiy by fixing the other entity' position"""
        cx, cy = 16, 12 # center
        half_x, half_y = 8, 8
        door_length = 3
        x_range = (cx - half_x, cx + half_x)
        y_range = (cy - half_y, cy + half_y)
        ref_pos_list = [(cx, cy), (cx + 6, cy - 6), (cx - 6, cy - 6), (cx - 6, cy + 6), ((cx + 6, cy + 6))]
        for ref_pos in ref_pos_list:
            trajs = [None, None]
            num_points = 0
            i = 1 
            while dx * i + x_range[0] < x_range[1]:
                j = 1
                while dy * j + y_range[0] < y_range[1]:
                    if dist((dx * i + x_range[0], dy * j + y_range[0]), ref_pos) < 1e-3:
                        j += 1
                        continue 
                    if not num_points:
                        trajs[entity_id] = [(dx * i + x_range[0], dy * j + y_range[0])]
                        trajs[1 - entity_id] = [ref_pos]
                    else:
                        trajs[entity_id].append((dx * i + x_range[0], dy * j + y_range[0]))
                        trajs[1 - entity_id].append(ref_pos)
                    num_points += 1
                    j += 1
                i += 1
            vels = [[(0, 0)] * num_points for _ in range(2)]
            segs = [
                    [(cx - half_x, cy + half_y), (cx + half_x, cy + half_y)],
                    [(cx + half_x, cy + half_y), (cx + half_x, cy - half_y)],
                    [(cx + half_x, cy - half_y), (cx - half_x, cy - half_y)],
                    [(cx - half_x, cy - half_y), (cx - half_x, cy + half_y - door_length)]
                    ]

            gen_coords = self.gen_coord_pool(trajs, vels, segs)
            print(self.model_path)
            model = pickle.load(open(self.model_path, 'rb'))
            selected, reg, mask = model['selected'], model['reg'], model['mask']
            print(reg.coef_)
            forces = self.pred_forces(reg, mask, gen_coords, selected, entities=[entity_id], remove_last=False)

            X = np.array([p[0] for p in trajs[entity_id]])
            Y = np.array([p[1] for p in trajs[entity_id]])
            # print(forces.shape)
            # print(num_points)
            Fx = np.array([forces[0, t, 0] for t in range(num_points)])
            Fy = np.array([forces[0, t, 1] for t in range(num_points)])

            plt.quiver(X, Y, Fx, Fy)
            plt.scatter(ref_pos[0], ref_pos[1], s=500)
            for seg in segs:
                plt.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], linewidth=5, color='black')
            # print(np.vstack([X,Y,Fx,Fy]))
            plt.axis('equal')
            plt.axis('off')
            plt.tight_layout()
            plt.show()


    def visualize_potential(self, dx=1.5, dy=1.5, entity_id=0, show_ref=True):
        """visualize the potentials for an entitiy by fixing the other entity' position"""
        cx, cy = 16, 12 # center
        half_x, half_y = 8, 8
        door_length = 3
        x_range = (cx - half_x, cx + half_x)
        y_range = (cy - half_y, cy + half_y)
        # x_list = np.arange(x_range[0], x_range[1], dx)
        # y_list = np.arange(y_range[0], y_range[1], dy)
        # width, height = len(x_list), len(y_list)
        ref_pos_list = [(cx, cy), (cx + 6, cy - 6), (cx - 6, cy - 6), (cx - 6, cy + 6), ((cx + 6, cy + 6))]
        for ref_pos in ref_pos_list:
            trajs = [None, None]
            num_points = 0
            width = 0
            height = 0
            i = 1 
            while dx * i + x_range[0] < x_range[1]:
                j = 1
                while dy * j + y_range[0] < y_range[1]:
                    if dist((dx * i + x_range[0], dy * j + y_range[0]), ref_pos) < 1e-3:
                        j += 1
                        continue 
                    if not num_points:
                        trajs[entity_id] = [(dx * i + x_range[0], dy * j + y_range[0])]
                        trajs[1 - entity_id] = [ref_pos]
                    else:
                        trajs[entity_id].append((dx * i + x_range[0], dy * j + y_range[0]))
                        trajs[1 - entity_id].append(ref_pos)
                    num_points += 1
                    j += 1
                    width = max(width, j)
                i += 1
                height = max(height, i)
            vels = [[(0, 0)] * num_points for _ in range(2)]
            segs = [
                    [(cx - half_x, cy + half_y), (cx + half_x, cy + half_y)],
                    [(cx + half_x, cy + half_y), (cx + half_x, cy - half_y)],
                    [(cx + half_x, cy - half_y), (cx - half_x, cy - half_y)],
                    [(cx - half_x, cy - half_y), (cx - half_x, cy + half_y - door_length)]
                    ]

            gen_coords = self.gen_coord_pool(trajs, vels, segs)
            print(self.model_path)
            model = pickle.load(open(self.model_path, 'rb'))
            selected, reg, mask = model['selected'], model['reg'], model['mask']
            print(reg.coef_)
            potentials = self.pred_potentials(reg, mask, gen_coords, selected, entities=[entity_id], remove_last=False)

            X = np.array([p[0] for p in trajs[entity_id]])
            Y = np.array([p[1] for p in trajs[entity_id]])
            print(potentials.shape)
            V = -np.array([potentials[0, t] for t in range(num_points)])
            
            p_id = 0
            i = 1
            Vmat = np.zeros((height - 1, width - 1))
            while dx * i + x_range[0] < x_range[1]:
                j = 1
                while dy * j + y_range[0] < y_range[1]:
                    Vmat[j - 1][i - 1] = -potentials[0, p_id]
                    if dist((dx * i + x_range[0], dy * j + y_range[0]), ref_pos) < 1e-3:
                        j += 1
                        continue 
                    p_id += 1
                    j += 1
                i += 1


            # Vmat = np.zeros((height, width))
            # for p_id in range(num_points):
            #     X, Y = trajs[entity_id][p_id][0], trajs[entity_id][p_id][1]
            #     V = -potentials[0, p_id]
            #     Vmat[int((X - x_range[0]) // dx), int((Y - y_range[0]) // dy)] = V
            #     print(int((X - x_range[0]) // dx), int((Y - y_range[0]) // dy))

            # plt.pcolormesh(x_list, y_list, Vmat)
            # plt.imshow(Vmat)

            fig, ax = plt.subplots()
            im = ax.imshow(Vmat, interpolation='bilinear', origin='lower', extent=x_range + y_range)

            forces = self.pred_forces(reg, mask, gen_coords, selected, entities=[entity_id], remove_last=False)
            X = np.array([p[0] for p in trajs[entity_id]])
            Y = np.array([p[1] for p in trajs[entity_id]])
            Fx = np.array([forces[0, t, 0] for t in range(num_points)])
            Fy = np.array([forces[0, t, 1] for t in range(num_points)])
            plt.quiver(X, Y, Fx, Fy)

            if show_ref:
                plt.scatter(ref_pos[0], ref_pos[1], s=500, c='red')
            for seg in segs:
                plt.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], linewidth=5, color='black')
            plt.axis('equal')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

