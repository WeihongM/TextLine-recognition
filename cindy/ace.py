# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from torch.autograd import Variable
from cindy.utils.basic import cal_distance
import torch
import pdb
import itertools
from warpctc_pytorch import CTCLoss
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import time
import math
import edit_distance as ed
from cindy.seq_module import Sequence

np.set_printoptions(precision=4,suppress=True,threshold=10000)



class ACE(Sequence):
    def __init__(self, class_num, para = 1.0):
        super(ACE, self).__init__()
        self.softmax = None;
        self.label = None;
        self.class_num = class_num+1
        self.out = nn.Linear(512, class_num+1)

        self.log_num = 0 # which picture logging

    def forward(self, input, label):
        input = self.out(input)
        input = F.softmax(input,dim=-1)

        # loss_ACE = torch.sum(input)
        Ts, bs, dim = input.size()



        num = Ts # num can be smaller that height*width
        ori_num = Ts # ori_num must be height*width

        # input_argmax = torch.argmax(input, dim=2)
        # qualify = input_argmax == self.class_num-1
        # self.qualify = input_argmax * qualify.type(torch.cuda.LongTensor)
        # qualify = qualify.type(torch.cuda.FloatTensor)
        # unqualify = 1- qualify

        # unqualify_sum = torch.sum(unqualify,0)
        # inf_flag = unqualify_sum == Ts
        # qualify[:,inf_flag] = 1
        # unqualify = 1- qualify

        # qualify = qualify.detach()
        # unqualify = unqualify.detach()
        # new1 = input[:,:,0:1] + input[:,:,-1:]*unqualify.unsqueeze(2)
        # new2 = input[:,:,-1:]*qualify.unsqueeze(2)
        # input = torch.cat([new1,input[:,:,1:-1], new2],-1)

        space = input[:,:,-1]
        space = space[1:,:]-space[:-1,:]
        space = torch.relu(space)
        space = torch.sum(space, 0)
        self.space = space

        # get init result of ACE
        self.softmax = input
        input = input.permute(1,2,0);
        sum_result = torch.sum(input,2) 
        sum_result[:,-1] =  space
        # sum_result[:,-1] =  sum_result[:,-1] - space
        sum_result[:,0] = sum_result[:,0] + (sum_result[:,-1] - space)


        # ************************************************************************************************
        # Normal ACE | 1D-CTC-like 
        # @orange
        # ************************************************************************************************
        batch_size, total_len = label.size()
        self.label = label
        label_array = np.zeros((batch_size, self.class_num))
        label_array[:,0]=ori_num
        for bn in range(batch_size):
            for tn in range(total_len):
                if label[bn][tn] != -1:
                    label_array[bn][int(label[bn][tn]+1)] += 1
                    label_array[bn,0] -= 1

        self.blank_pred = sum_result[:,-1]
        self.blank_label = label_array[:,-1]

        sum_result = sum_result/num
        label_array = label_array/num
        label_var = Variable(torch.from_numpy(label_array).type(torch.cuda.FloatTensor), requires_grad=False).cuda()
        # print(unqualify_sum)
        # print(torch.sum(torch.log(sum_result)*label_var, -1))
        loss_ACE =  (-torch.sum(torch.log(sum_result)*label_var))/batch_size # original ACE
        return loss_ACE

    def decode_batch(self):
        length,batch_size,dim = self.softmax.shape
        out_best = torch.max(self.softmax, 2)[1].data
        out_best = out_best.cpu().numpy()
        out_best = out_best.reshape(-1,batch_size)
        out_best = out_best.transpose()

        out_best_list = [0]*batch_size
        for j in range(out_best.shape[0]):
            # out_best_new = [k for k, g in itertools.groupby(out_best[j])]
            out_best_new = out_best[j]
            out_best_list[j] = [int(x) for x in out_best_new if x != 0]

        if self.training or np.random.rand() < 0.02:
            alphabet = '_$!$#$"$\'$&$)$($+$*$-$,$/$.$1$0$3$2$5$4$7$6$9$8$;$:$?$A$C$B$E$D$G$F$I$H$K$J$M$L$O$N$Q$P$S$R$U$T$W$V$Y$X$Z$a$c$b$e$d$g$f$i$h$k$j$m$l$o$n$q$p$s$r$u$t$w$v$y$x$z$|'.split('$')
            print(''.join([alphabet[i] for i in out_best[0]]))
            # print(''.join([alphabet[i] for i in self.qualify[:,0]]))
            print('pred :',''.join([alphabet[i] for i in out_best_list[0]]))
            label_list = self.label[0][self.label[0]!=-1].tolist()
            print('label:',''.join([alphabet[i+1] for i in label_list]))
            print('blank_pred :', self.blank_pred.data.cpu().numpy())
            print('blank_label:', self.blank_label.astype('float'))

        return out_best_list