from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import random
import pickle
import copy
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from utils import *


np.set_printoptions(precision=3, suppress=True)
RADIUS = 1.0
MASS   = math.pi * (RADIUS ** 2)
points = [(8, 20), (8, 20 - 3), (8, 4), (24, 4), (24, 20)] # reference points


class GenCoordModel_Physical():
    """
    Learning generalized coordinates and potential functions
    """
    def __init__(self, model_path=None):
        self.model_path = model_path


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
                    # print(after_dist)
                    if after_dist < RADIUS * 2:
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


    def prepare_data(self, gen_coords, Fs=None, selected=None):
        feat_X, feat_Y  = {}, {}
        if Fs:
            gt_X, gt_Y = [], []
            for gen_coords_one_episode, forces in zip(gen_coords, Fs):
                sources = gen_coords_one_episode['sources']
                gen_coords_all = gen_coords_one_episode['gen_coords']
                N = gen_coords_one_episode['num_entities']
                for gen_coord_id, (source, qs) in enumerate(zip(sources, gen_coords_all)):
                    if selected and gen_coord_id not in selected: continue
                    cur_feat_X, cur_feat_Y = get_feat(source, qs, N)
                    if gen_coord_id not in feat_X:
                        feat_X[gen_coord_id], feat_Y[gen_coord_id] = [], []
                    for entity_id in range(N):
                        feat_X[gen_coord_id].append(cur_feat_X[entity_id])
                        feat_Y[gen_coord_id].append(cur_feat_Y[entity_id])
                for entity_id in range(N):
                    gt_X.append(np.vstack([np.array(f[0]) for f in forces[entity_id]])) 
                    gt_Y.append(np.vstack([np.array(f[1]) for f in forces[entity_id]]))
            return feat_X, feat_Y, gt_X, gt_Y
        else:
            for gen_coords_one_episode in gen_coords:
                sources = gen_coords_one_episode['sources']
                gen_coords_all = gen_coords_one_episode['gen_coords']
                N = gen_coords_one_episode['num_entities']
                for gen_coord_id, (source, qs) in enumerate(zip(sources, gen_coords_all)):
                    cur_feat_X, cur_feat_Y = get_feat(source, qs, N)
                    if gen_coord_id not in feat_X:
                        feat_X[gen_coord_id], feat_Y[gen_coord_id] = [], []
                    for entity_id in range(N):
                        feat_X[gen_coord_id].append(cur_feat_X[entity_id])
                        feat_Y[gen_coord_id].append(cur_feat_Y[entity_id])
            return feat_X, feat_Y


    def pred_forces(self, reg, gen_coords, selected):
        """pred forces given selected generalized coordinates and the fitted model"""
        feat_X, feat_Y = self.prepare_data([gen_coords])
        feat_X = np.concatenate([np.vstack(feat_X[sel_id]) for sel_id in selected], axis=-1)
        feat_Y = np.concatenate([np.vstack(feat_Y[sel_id]) for sel_id in selected], axis=-1)
        forces = np.concatenate([reg.predict(feat_X).reshape(-1, 1), 
                                 reg.predict(feat_Y).reshape(-1, 1)], axis=-1)
        """TODO: different #entities"""
        return forces.reshape(gen_coords['num_entities'], -1, 2)


    def pred_vels(self, reg, gen_coords, selected, vels):
        """pred velocities at next time step given selected generalized coordinates and the fitted model"""
        forces = self.pred_forces(reg, gen_coords, selected)
        accs = forces / MASS
        pred = np.array(vels)
        pred[:, 1:, :] = pred[:, :-1, :] + accs
        return pred


    def train(self, data, train_size):
        """TODO: currently assume same training scenarios (#entities and environment)"""
        trajs, vels, Fs, segs = data['trajs'], data['vels'], data['Fs'], data['segs']
        num_episodes = min(len(trajs), train_size)
        trajs, vels, Fs, segs = trajs[:train_size], vels[:train_size], Fs[:train_size], segs[:train_size]
        
        #accs = [[get_accs(vel, 1) for vel in vel_2] for vel_2 in vels]
        #Fs = [[get_Fs(acc, MASS) for acc in acc_2] for acc_2 in accs]
        
        f_gt_list = [np.stack([np.vstack([np.array(f) for f in forces]) \
                                        for forces in Fs[episode_id]]) \
                            for episode_id in range(num_episodes)]
        f_gt = np.vstack(f_gt_list)
        vel_gt_list = [np.stack([np.vstack([np.array(v) for v in vel]) \
                                            for vel in vels[episode_id]]) \
                            for episode_id in range(num_episodes)]
        vel_gt = np.vstack(vel_gt_list)
        gen_coords = [self.gen_coord_pool(trajs[episode_id], vels[episode_id], segs[episode_id]) \
                            for episode_id in range(num_episodes)]
        feat_X, feat_Y, gt_X, gt_Y = self.prepare_data(gen_coords, f_gt_list)
        gt = np.vstack(gt_X + gt_Y)
        sources = gen_coords[0]['sources']
        types = gen_coords[0]['types']

        selected = []
        selected_types = []
        final_reg = None
        cur_loss = 1e8
        f_loss_list = [mean_squared_error(f_gt.reshape(-1, 2), np.zeros(f_gt.reshape(-1, 2).shape))]
        v_loss_list = [mean_squared_error(vel_gt.reshape(-1, 2), np.zeros(vel_gt.reshape(-1, 2).shape))]
        for iter_id in range(20):
            min_loss, selected_id, selected_reg = None, None, None
            gen_coord_list = range(len(sources))
            for gen_coord_id in gen_coord_list:
                if gen_coord_id in selected: continue
                if not feat_X[gen_coord_id] or not feat_Y[gen_coord_id]: continue
                cur_selected = list(selected) + [gen_coord_id]
                feat = np.concatenate([np.vstack(feat_X[sel_id] + feat_Y[sel_id]) \
                                    for sel_id in cur_selected], axis=-1)

                num_samples = feat.shape[0]
                N_train = num_samples
                X_train, Y_train = feat[:N_train, :], gt[:N_train, :].reshape(-1)

                # fit
                # print(gen_coord_id, sources[gen_coord_id])
                # print(X_train.shape, Y_train.shape)
                #for i in range(N_train):
                #    if abs(X_train[i, 0]) > 1e-8:
                #        print(X_train[i], Y_train[i], Y_train[i] / X_train[i, 0]) 
                # alpha = 0.0002
                # reg = Lasso(alpha=alpha, fit_intercept=False).fit(X_train, Y_train)
                reg = Ridge(alpha=0.01, fit_intercept=False).fit(X_train, Y_train)
                # reg = LinearRegression(fit_intercept=False).fit(X_train, Y_train)
                # print(reg.coef_)
                Y_pred = reg.predict(X_train)
                err = Y_pred - Y_train
                mean_err = err.mean()
                std_err = err.std()

                X_train = X_train[abs(err - mean_err) < std_err * 2, :]
                Y_train = Y_train[abs(err - mean_err) < std_err * 2]
                # reg = Lasso(alpha=alpha, fit_intercept=False).fit(X_train, Y_train)
                reg = Ridge(alpha=0.01, fit_intercept=False).fit(X_train, Y_train)
                # reg = LinearRegression(fit_intercept=False).fit(X_train, Y_train)
                
                # predict
                f_pred_list = [self.pred_forces(reg, gen_coords[episode_id], cur_selected)
                                for episode_id in range(num_episodes)]
                f_pred = np.vstack(f_pred_list)
                loss = mean_squared_error(f_gt.reshape(-1, 2), f_pred.reshape(-1, 2))
                # print(f_gt.shape, f_pred.shape, loss)
                # input('press any key to continue...')

                if min_loss is None or loss < min_loss - 1e-8:
                    min_loss = loss
                    selected_id = gen_coord_id
                    selected_reg = reg
                # print('loss:', loss)
                # print('params:', reg.coef_, reg.intercept_)
                # input('press any key to continue...')
            if selected_id is None or min_loss > cur_loss - 0.001: break
            selected.append(selected_id)
            selected_types.append(types[selected_id])
            final_reg = selected_reg
            cur_loss = min_loss
            f_loss_list.append(cur_loss)
            print('iter %d:  #%d gen coord (%d %d), loss %.3f' \
                % (iter_id, selected_id, sources[selected_id][0], sources[selected_id][1],
                   cur_loss))
            print('params:', final_reg.coef_)

            vel_pred_list = [self.pred_vels(final_reg, gen_coords[episode_id], selected, vels[episode_id])
                                for episode_id in range(num_episodes)]
            vel_pred = np.vstack(vel_pred_list)
            loss = mean_squared_error(vel_gt.reshape(-1, 2), vel_pred.reshape(-1, 2))
            print('loss (velocities):', loss)
            v_loss_list.append(loss)

            # input('press any key to continue...')
        if self.model_path:
            pickle.dump({'selected': selected, 'types': selected_types, 'reg': final_reg, 
                         'f_loss_list': f_loss_list, 'vel_loss_list': v_loss_list},
                        open(self.model_path, 'wb'))
        else:
            self.res = {'selected': selected, 'types': selected_types, 'reg': final_reg, 
                         'f_loss_list': f_loss_list, 'vel_loss_list': v_loss_list}


    def test(self, data):
        """test the model"""
        model = pickle.load(open(self.model_path, 'rb'))
        selected, reg = model['selected'], model['reg']

        trajs, vels, Fs, segs = data['trajs'], data['vels'], data['Fs'], data['segs']
        num_episodes = len(trajs)
        #vels = [self.adj_vels(trajs[episode_id], vels0[episode_id], segs[episode_id])
        #            for episode_id in range(num_episodes)]
        f_gt_list = [np.stack([np.vstack([np.array(f) for f in forces]) \
                                        for forces in Fs[episode_id]]) \
                            for episode_id in range(num_episodes)]
        f_gt = np.concatenate([f.reshape(-1, 2) for f in f_gt_list], 0)
        gen_coords = [self.gen_coord_pool(trajs[episode_id], vels[episode_id], segs[episode_id]) \
                            for episode_id in range(num_episodes)]

        """TODO: different sources in different episodes"""
        sources = gen_coords[0]['sources']

        # f_pred_list = [self.pred_forces(reg, gen_coords[episode_id], selected)
        #                         for episode_id in range(num_episodes)]
        # f_pred = np.concatenate([f.reshape(-1, 2) for f in f_pred_list], 0)
        # loss = mean_squared_error(f_gt.reshape(-1, 2), f_pred.reshape(-1, 2))
        # print('Testing loss (forces):', loss)
        
        vel_pred_list = [self.pred_vels(reg, gen_coords[episode_id], selected, vels[episode_id])
                                for episode_id in range(num_episodes)]
        vel_pred = np.concatenate([v.reshape(-1, 2) for v in vel_pred_list], 0)
        vel_gt_list = [np.stack([np.vstack([np.array(v) for v in vel]) \
                                        for vel in vels[episode_id]]) \
                            for episode_id in range(num_episodes)]
        vel_gt = np.concatenate([v.reshape(-1, 2) for v in vel_gt_list], 0)
        loss = []
        for A, B in zip(vel_gt_list, vel_pred_list):
            T = A.shape[1]
            # print(T)
            N = A.shape[0]
            # ave_vel = np.linalg.norm(A, axis=-1).mean(-1)
         #   input('press any key to continue...')
            loss.append([[mean_squared_error(A[i,t,:], B[i,t,:]) for t in range(T)] 
                            for i in range(N)])
           # print(max(loss[-1][0]), max(loss[-1][1]), sum(loss[-1][0]) / len(loss[-1][0]), 
           #         sum(loss[-1][1]) / len(loss[-1][1]))
        print('Testing loss (velocities):', mean_squared_error(vel_gt.reshape(-1, 2), vel_pred.reshape(-1, 2)))
        return loss
        

