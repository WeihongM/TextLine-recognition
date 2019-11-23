# -*- coding: utf-8 -*-
from __future__ import print_function, division


from cindy.seq_module import *
from cindy.ace import ACE
from cindy.ctc import CTC
from cindy.attention import BaiAttention
from cindy.solver import seq_solver

import settings
torch.manual_seed(1671)
torch.cuda.manual_seed_all(1671)

def showsettings(s):
    for key in s.keys():
        print(key , s[key])
    print('')

if __name__ == "__main__":

    showsettings(settings.optimizer)
    showsettings(settings.scheduler)
    showsettings(settings.others)
    showsettings(settings.data_args)

    the_solver = seq_solver(criterion = CTC, 
                        model = settings.model,
                        lmdb = settings.lmdb,
                        optimizer = settings.optimizer, 
                        scheduler = settings.scheduler, 
                        train_solver = 'seq_solver',
                        others = settings.others)

    the_solver.forward()