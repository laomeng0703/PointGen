#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 

"""
import os
import pprint
pp = pprint.PrettyPrinter()
from datetime import datetime

from train_model import Model
from config import args


if __name__ == '__main__':
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    args.log_dir = os.path.join(args.log_dir, args.model_name+"_cls", current_time)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    print('checkpoints:', args.log_dir)

    model = Model(args)
    model.train()

