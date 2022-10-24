#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
"""
@file: augmentor.py
@description: 
"""
import os
import sys
import copy
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
import random


def knn(x, k):                                              # distance = (x - x_train)^2 = x^2 - 2*x*x_train + x_train^2
    inner = -2 * torch.matmul(x.transpose(2, 1), x)         # -2*x*x_train
    xx = torch.sum(x ** 2, dim=1, keepdim=True)             # x^2
    pairwise_distance = -xx - inner - xx.transpose(2, 1)    # x^2 - 2*x*x_train + x_train^2

    idx = pairwise_distance.topk(k=k, dim=-1)[1]            # (batch_size, num_points, k)，返回最近的k个元素的索引值
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size()[0]
    num_points = x.size()[2]
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  # (batch_size, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)
    x = x.transpose(2, 1).contiguous()
    # -> (batch_size*num_points, num_dims) -> (batch_size * num_points * k + range(0, batch_size*num_points))
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) # (b, n, k, c)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) # (b, n, k, c)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous() # (b, 2*c, n, k)

    return feature  # (batch_size, 2*num_dims, num_points, k)

def batch_quat_to_rotmat(q, out=None):

    B = q.size(0)

    if out is None:
        out = q.new_empty(B, 3, 3)

    # 2 / squared quaternion 2-norm
    len = torch.sum(q.pow(2), 1)
    s = 2/len

    s_ = torch.clamp(len,2.0/3.0,3.0/2.0)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = (1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s))#.mul(s_)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = (1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s))#.mul(s_)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = (1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s))#.mul(s_)

    return out, s_
    
class Generator_Rotation(nn.Module):
    def __init__(self, dim):
        super(Generator_Rotation, self).__init__()

        self.fc1 = nn.Linear(dim + 1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size()[0]                         # (batch_size, emb_dims+noise_dims)
        x = F.relu(self.bn1(self.fc1(x)))       # (batch_size, 512)
        x = F.relu(self.bn2(self.fc2(x)))       # (batch_size, 256)
        x = self.fc3(x)                         # (batch_size, 4)

        # iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(B, 1)
        # if x.is_cuda:
        #     iden = iden.cuda()
        # x = x + iden
        # x = x.view(-1, 3, 3)

        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        # convert quaternion to rotation matrix
        x, s = batch_quat_to_rotmat(x)
        x = x.view(-1, 3, 3)
        s = s.view(B, 1, 1)
        return x, None


class Generator_Displacement(nn.Module):
    def __init__(self, dim):
        super(Generator_Displacement, self).__init__()
        self.conv1 = torch.nn.Conv1d(dim+1024+512, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 3, 1)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)                               # (batch_size, 3, num_points)

        return x

class Generator(nn.Module):
    def __init__(self, args, dim=1024, in_dim=6):
        super(Generator, self).__init__()
        self.dim = dim
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        # parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')

        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, 64, kernel_size=1, bias=False,),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.rot = Generator_Rotation(args.emb_dims)
        self.dis = Generator_Displacement(args.emb_dims)

    def forward(self, pt, noise):
        B, C, N = pt.size()
        raw_pt = pt[:, :3, :].contiguous()

        normal = pt[:, 3:, :].transpose(1, 2).contiguous() if C > 3 else None

        x = get_graph_feature(raw_pt, k=self.k)             # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                                   # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = torch.max(x, dim=-1, keepdim=False)[0]         # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)                 # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                                   # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = torch.max(x, dim=-1, keepdim=False)[0]         # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)                 # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                                   # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = torch.max(x, dim=-1, keepdim=False)[0]         # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)                 # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                                   # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = torch.max(x, dim=-1, keepdim=False)[0]         # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        pointfeat = torch.cat((x1, x2, x3, x4), dim=1)      # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(pointfeat)                           # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x_pool1 = F.adaptive_max_pool1d(x, 1).view(B, -1)   # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(B, -1)        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)

        #feat_r = torch.cat((x1, x2), 1)                    # (batch_size, emb_dims*2)
        feat_r = x_pool1                                    # (batch_size, emb_dims)
        feat_r = torch.cat([feat_r, noise], 1)              # (batch_size, emb_dims + noise_dim)
        rotation, scale = self.rot(feat_r)                  # (batch_size, 3, 3), (batch_size)

        feat_d = x_pool1.view(-1, self.dim, 1).repeat(1, 1, N)   # (batch_size, emb_dims, 1) -> (batch_size, emb_dims, num_points)
        noise_d = noise.view(B, -1, 1).repeat(1, 1, N)

        feat_d = torch.cat((pointfeat, feat_d, noise_d), 1) # (batch_size, 512+emb_dims+noise_dims, num_points)
        displacement = self.dis(feat_d)                     # (batch_size, 3, num_points)

        pt = raw_pt.transpose(2, 1).contiguous()            # (batch_size, num_points, 3)

        p1 = random.uniform(0, 1)
        possible = 0.5
        if p1 > possible:
            pt = torch.bmm(pt, rotation).transpose(1, 2).contiguous()   # (batch_size, 3, num_points)
        else:
            pt = pt.transpose(1, 2).contiguous()                        #
        
        p2 = random.uniform(0, 1)
        if p2 > possible:
            pt = pt + displacement                                      # (batch_size, 3, num_points)

        if normal is not None:
            normal = (torch.bmm(normal, rotation)).transpose(1, 2).contiguous()
            pt = torch.cat((pt, normal), 1)

        return pt


if __name__ == '__main__':
    from config import args
    pt = torch.randn((24, 3, 1024)).cuda()
    noise = torch.randn((24, 1024)).cuda()

    net = Generator(args).cuda()
    points = net(pt, noise)
    print('output point size:', points.shape)



