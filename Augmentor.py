

import torch
import torch.nn as nn
import numpy as np

class Augmentor():
    """
    Pointcloud augmentation with local weighted transformation
    """
    def __init__(self, args):
        self.num_anchor = args.w_num_anchor
        self.sample_type = args.w_sample_type
        self.sigma = args.w_sigma                                       # bandwidth, h

        self.R_range = (-abs(args.w_R_range), abs(args.w_R_range))      # rotation range
        self.S_range = (1., args.w_S_range)                             # scaling range
        self.T_range = (-abs(args.w_T_range), abs(args.w_T_range))      # translation range

    def __call__(self, pos):
        """
        input:
            pos: original pointcloud, [N, 3]

        output:
            pos: [N, 3]
            pos_new: Augmented pointcloud, [N, 3]
        """
        M = self.num_anchor                                             # number of anchor points
        N, _ = pos.shape                                                # number of points

        if self.sample_type == 'fps':
            idx = self.fps(pos, M)
        else:
            idx = np.random.choice(N, M)

        pos_anchor = pos[idx]                                           # (M, 3), anchor points

        pos_r = np.expand_dims(pos, 0).repeat(M, axis=0)                # (M, N, 3)

        # move to canonical space
        pos_normalize = pos_r - pos_anchor.reshape(M, -1, 3)            # (M, N, 3)

        # local transform
        pos_transformed = self.local_transform(pos_normalize)           # (M, N, 3)

        # move to origin space
        pos_transformed = pos_transformed + pos_anchor.reshape(M, -1, 3)# (M, N, 3)

        # kernel regression
        pos_new = self.kernel_regression(pos, pos_anchor, pos_transformed)

        # normalize
        pos_new = self.normalize(pos_new)

        return pos.astype('float32'), pos_new.astype('float32')


    def fps(self, pos, npoint):
        """
        input:
            pos: pointcloud data, [N, 3]
            npoint: number of samples [int]

        return:
            centroids: sampled pointcloud index, [npoint]
        """
        N, _ = pos.shape

        centroids = np.zeros(npoint, dtype=np.int_)                     # M, anchor points
        distance = np.ones(N, dtype=np.float64) * 1e10                  # N, 采样点到所有点距离
        farthest = np.random.randint(0, N, (1,), dtype=np.int_)         # 初始时随机选择一点

        for i in range(npoint):
            centroids[i-1] = farthest                                     # 更新第i个最远点
            centroids = pos[farthest, :]                                # 取出这个最远点的xyz坐标
            dist = ((pos - centroids) ** 2).sum(-1)                     # 计算点集中的所有点到这个最远点的欧式距离
            mask = dist < distance
            distance[mask] = dist[mask]                                 # 更新distance，记录样本中每个点距离所有已出现的采样点的最小距离
            farthest = distance.argmax()                                # 返回最远点索引
        return centroids

    def local_transform(self, pos_normalize):
        """
        input:
            pos_normalize: points in canonical space, [M, N, 3]

        output:
            pos_transformed: Pointclouds after local transformation centered at M anchor points, [M, N, 3]
        """
        M, N, _ = pos_normalize.shape
        transformation_dropout = np.random.binomial(1, 0.5, (M, 3))     # binomial distribution
        transformation_axis = self.get_random_axis(M)

        # randomly sampling from (-R_range, R_range)
        degree = np.pi * np.random.uniform(*self.R_range, size=(M, 3)) / 180.0 * transformation_dropout[:, 0:1]
        # randomly sampling from (1, S_range)
        scale = np.random.uniform(*self.S_range, size=(M, 3)) * transformation_dropout[:, 1:2]
        scale = scale * transformation_axis
        scale = scale + 1*(scale==0)                                    # scaling factor must bigger than 1.
        # randomly sampling from (1, T_range)
        translation = np.random.uniform(*self.T_range, size=(M, 3)) * transformation_dropout[:, 2:3]
        translation = translation * transformation_axis

        # scaling matrix
        S = np.expand_dims(scale, axis=1) * np.eye(3)                   # turn scaling factor to diagonal matrix, (M, 3)->(M, 3, 3)
        # rotation matrix
        sin = np.sin(degree)
        cos = np.cos(degree)
        sx, sy, sz = sin[:, 0], sin[:, 1], sin[:, 2]
        cx, cy, cz = cos[:, 0], cos[:, 1], cos[:, 2]
        R = np.stack([cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx,
                      sz*cy, sz*sy*sx + cz*cy, sz*sy*cx - cz*sx,
                      -sy, cy*sx, cy*cx], axis=1).reshape(M, 3, 3)
        #R = np.expand_dims(R, axis=0).repeat(M, axis=0)

        pos_transformed = pos_normalize @ R @ S + translation.reshape(M, 1, 3)
        return pos_transformed

    def get_random_axis(self, n_axis):
        """
        input:
            n_axis: number of axis

        output:
            axis: projection axis, [n_axis, 3]
        """
        axis = np.random.randint(1, 8, (n_axis)) # 1(001):z, 2(010):y, 3(011):yz, 4(100):x, 5(101):xz, 6(110):xy, 7(111):xyz
        m = 3
        axis = (((axis[:, None] & (1 << np.arange(m)))) > 0).astype(int)
        return axis

    def normalize(self, pos):
        """
        input:
            pos: pointcloud, [N, 3]

        output:
            pos: normalized pointcloud, [N, 3]
        """
        pos = pos - pos.mean(axis=-2, keepdims=True)
        scale = (1 / np.sqrt((pos ** 2).sum(1)).max()) * 0.999999
        pos = scale * pos
        return pos

    def kernel_regression(self, pos, pos_anchor, pos_transformed):
        """
        input:
            pos: pointcloud, [N, 3]
            pos_anchor: anchor points, [M, 3]
            pos_transformed: Pointclouds after local transformation centered at M anchor points, [M, N, 3]

        output:
            pos_new: pointcloud after weighted local transformation, [N, 3]
        """
        M, N, _ = pos_transformed.shape

        # Distance between anchor points & entire points
        sub = np.expand_dims(pos_anchor, axis=1).repeat(N, axis=1) - np.expand_dims(pos, axis=0).repeat(M, axis=0)

        project_axis = self.get_random_axis(1)
        projection = np.expand_dims(project_axis, axis=1) * np.eye(3)   # projection matrix, (1, 3, 3)

        # project distance
        sub = sub @ projection                                          # (M, N, 3) dot (1, 3, 3) -> (M, N, 3)
        sub = np.sqrt(((sub) ** 2).sum(2))                              # (M, N)

        # kernel regression
        weight = np.exp(-0.5 * (sub ** 2) / (self.sigma ** 2))
        pos_new = (np.expand_dims(weight, axis=2).repeat(3, axis=2) * pos_transformed).sum(0)
        pos_new = (pos_new / weight.sum(0, keepdims=True).T)            # normalized by weight

        return pos_new
