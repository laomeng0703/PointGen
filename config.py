#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 

"""

import argparse
import os

def str2bool(x):
    return x.lower() in ('true')

parser = argparse.ArgumentParser('PointNet')
parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
parser.add_argument('--epoch',  default=250, type=int, help='number of epoch in training')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
parser.add_argument('--learning_rate_a', default=0.001, type=float, help='learning rate in training')
parser.add_argument('--no_decay', type=str2bool, default=False)
parser.add_argument('--noise_dim',  default=1024, type=int, help='dimension of noise')

parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
parser.add_argument('--pretrain', type=str, default=None, help='whether use pretrain Augment')
parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate of learning rate')

parser.add_argument('--model_name', default='pointnet', help='classification model')
parser.add_argument('--log_dir', default='log', help='log_dir')
parser.add_argument('--data_dir', default='ModelNet40_Folder')
parser.add_argument('--epoch_per_save', type=int, default=5)
parser.add_argument('--num_points', type=int, default=1024)
parser.add_argument('--y_rotated', type=str2bool, default=True)
parser.add_argument('--augment', type=str2bool, default=False)
parser.add_argument('--use_normal', type=str2bool, default=False)
parser.add_argument('--restore', action='store_true')
parser.add_argument('--generator', action='store_true', default=True, help='Use new generator')

# Training settings

parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=20, metavar='N',
                    help='Num of nearest neighbors to use')
parser.add_argument('--model_path', type=str, default='', metavar='N',
                    help='Pretrained model path')

# local transform settings
parser.add_argument('--Augmentor', action='store_true', default=False, help='Use Augmentor')

parser.add_argument('--w_num_anchor', type=int, default=4, help='Num of anchor point')
parser.add_argument('--w_sample_type', type=str, default='fps',
                    help='Sampling method for anchor point, option : (fps, random)')
parser.add_argument('--w_sigma', type=float, default=0.5, help='Kernel bandwidth')

parser.add_argument('--w_R_range', type=float, default=10, help='Maximum rotation range of local transformation')
parser.add_argument('--w_S_range', type=float, default=3, help='Maximum scailing range of local transformation')
parser.add_argument('--w_T_range', type=float, default=0.25, help='Maximum translation range of local transformation')

# AugTune settings
# parser.add_argument('--AugTune', action='store_true', help='Use AugTune')
# parser.add_argument('--l', type=float, default=0.1, help='Difficulty parameter lambda')

args = parser.parse_args()
