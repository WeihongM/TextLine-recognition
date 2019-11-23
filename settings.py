# -*- coding: utf-8 -*-
from __future__ import print_function, division
from torch import optim

exp_name = {
    'exp_name'  :'dzj'
}

# Model setting
model = {
    'mode'          : "Train",
    'ngpu'          : [0],
    # 'model_type'    : ResnetEncoderDecoder,
    'model_type'    : "ChineseEncoderDecoder",
    'loading_epoch' : 0,
    'model_path'    :'model/' + exp_name['exp_name'] + '-{:0>2d}.pkl',
}

optimizer = {
    'optim_type'    : optim.SGD,
    'lr'            : 0.1,      # 1.0
    'momentum'      : 0.9,
    'weight_decay'  : 0.0001
}

scheduler = {
    'scheduler_type': optim.lr_scheduler.MultiStepLR,
    'milestones'    : [30, 40],
    'gamma'         : 0.1,
    'total_epoch'   : 50
}

others = {
    'log_path'  : 'log/' + exp_name['exp_name']
}

# DataLoader setting
lmdb = {
    'lmdb_train_type'  : "DZJDataset",
    'batch_size_train': 1, # actually 32
    'lmdb_test_type'  : "DZJDatasetTest",
    'batch_size_test': 1, # actually 1
}

data_args = {
    'AUG'       : True,
    'DropLabel' : False,
    'DropRatio' : 0.2,  # 1 - DropRatio are dropped
    'USE_SYNDATA': False,
    'SYN_RATIO' : 0.3,
    'Train_Batch': 32, # 3gpus
    'HEIGHT'    : 96,
}