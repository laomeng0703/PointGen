from __future__ import print_function
import os
import argparse
import torch

# from util import IOStream
# from train import train_vanilla, train_AugTune, test

# def _init_():
#     if not os.path.exists('checkpoints'):
#         os.makedirs('checkpoints')
#     if not os.path.exists('checkpoints/'+args.exp_name):
#         os.makedirs('checkpoints/'+args.exp_name)
#     if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
#         os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
#     os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
#     os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
#     os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
#     os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')
#     os.system('cp PointWOLF.py checkpoints' + '/' + args.exp_name + '/' + 'PointWOLF.py.backup')
#     os.system('cp train.py checkpoints' + '/' + args.exp_name + '/' + 'train.py.backup')


parser = argparse.ArgumentParser(description='Point Cloud Recognition')

parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--num_points', type=int, default=1024, help='number of points')
# local transformation settings
parser.add_argument('--Augmentor', action='store_true', help='Use local transform')

parser.add_argument('--w_num_anchor', type=int, default=4, help='Num of anchor point')
parser.add_argument('--w_sample_type', type=str, default='fps',
                    help='Sampling method for anchor point, option : (fps, random)')
parser.add_argument('--w_sigma', type=float, default=0.5, help='Kernel bandwidth')

parser.add_argument('--w_R_range', type=float, default=10, help='Maximum rotation range of local transformation')
parser.add_argument('--w_S_range', type=float, default=3, help='Maximum scailing range of local transformation')
parser.add_argument('--w_T_range', type=float, default=0.25,
                    help='Maximum translation range of local transformation')

args = parser.parse_args()