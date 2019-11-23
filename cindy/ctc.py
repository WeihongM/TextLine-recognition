# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from torch.autograd import Variable
from cindy.utils.basic import cal_distance
import torch
import itertools
from warpctc_pytorch import CTCLoss
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import edit_distance as ed
from cindy.seq_module import Sequence

np.set_printoptions(precision=4,suppress=True,threshold=10000)

class CTC(Sequence):
    def __init__(self):
        super(CTC, self).__init__()
        self.criterion = CTCLoss().cuda()
        self.softmax = None;
        self.label = None;
        self.len = 0;
        self.times = 0;
        self.count = 0;

    def forward(self, input, label, image):
        self.image = image.permute(0,2,3,1).data.cpu().numpy()
        self.softmax = input;
        self.label = label;

        batch_size, total_len = label.size()
        label_len = np.zeros(batch_size)
        label_seq = []
        label_total = 0;
        for bn in range(batch_size):
          for tn in range(total_len):
            if label[bn][tn] != -1:
              label_len[bn] = label_len[bn]+1
              label_total += 1
              label_seq.append(label[bn][tn]+1)
        label_seq = np.array(label_seq)
        label_len = Variable(torch.from_numpy(label_len).type(torch.IntTensor), requires_grad=False)
        label = Variable(torch.from_numpy(label_seq).type(torch.IntTensor), requires_grad=False)

        probs_sizes = Variable(torch.IntTensor([input.size(0)]*batch_size), requires_grad=False)
        loss = self.criterion(input, label, probs_sizes, label_len)/label_total
        return loss

    def decode_batch(self):
        length,batch_size,dim = self.softmax.shape
        out_best = torch.max(self.softmax, 2)[1].data
        out_best = out_best.cpu().numpy()
        out_best = out_best.reshape(-1,batch_size)
        out_best = out_best.transpose()

        out_best_list = [0]*batch_size
        for j in range(out_best.shape[0]):
            out_best_new = [k for k, g in itertools.groupby(out_best[j])]
            out_best_list[j] = [int(x) for x in out_best_new if x != 0]

        return out_best_list

    def result_analysis(self):
        prediction = self.decode_batch()

        batch_size = self.softmax.size(1)
        delete_total = 0
        replace_total = 0
        insert_total = 0
        len_total = 0
        word_total = 0
        for i in range(batch_size):
            pre_list = prediction[i]
            label_list = self.label[i][self.label[i]!=-1].tolist()
            label_list = [int(ele)+1 for ele in label_list]
            distance, (delete, replace, insert) = cal_distance(label_list, pre_list)
            delete_total += delete
            replace_total += replace
            insert_total += insert
            len_total += len(label_list)
            if distance == 0:
                word_total += 1
        result = [delete_total, replace_total, insert_total, len_total, word_total]
        # result = [delete_total, replace_total, insert_total, len_total]
        return prediction, result   