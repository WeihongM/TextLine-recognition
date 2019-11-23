import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from warpctc_pytorch import CTCLoss

from cindy.utils.basic import timeSince
from cindy.utils.basic import cindy_gradient_clip
from cindy.modules.model import *
from cindy.utils.dzj_data import DZJDataset
from cindy.utils.dzj_test_data import DZJDatasetTest

class solver(object):

    def __init__(self, criterion, model, lmdb, optimizer, scheduler, train_solver, others, pre_train = False):

        self.loss_layer = criterion()
        self.model = eval(model['model_type'])().cuda()
        self.mode = model['mode']

        self.multigpus = False

        if model['ngpu'] != None:
            self.multigpus = True

        if self.multigpus:
            self.model = nn.DataParallel(self.model, device_ids=model['ngpu'])
        print(self.model)

        if optimizer['momentum']:
            self.optimizer = optimizer['optim_type'](self.model.parameters(), lr=optimizer['lr'], momentum=optimizer['momentum'], weight_decay = optimizer['weight_decay'])
        else:
            self.optimizer = optimizer['optim_type'](self.model.parameters(), lr=optimizer['lr'], weight_decay = optimizer['weight_decay'])

        if self.mode == "Train":
            data_set = eval(lmdb['lmdb_train_type'])()
            self.lmdb_train = DataLoader(data_set, batch_size=lmdb['batch_size_train'], shuffle=True, num_workers=4)
            data_set = eval(lmdb['lmdb_test_type'])()
            self.lmdb_test = DataLoader(data_set, batch_size=lmdb['batch_size_test'], shuffle=False, num_workers=4)
        else:
            data_set = eval(lmdb['lmdb_test_type'])()
            self.lmdb_test = DataLoader(data_set, batch_size=lmdb['batch_size_test'], shuffle=False, num_workers=4)

        self.total_epoch = scheduler['total_epoch']
        self.others = others
        self.log_path = others['log_path']
        self.model_path = model['model_path']
        if train_solver == 'seq_solver':
            self.train_solver = seq_solver
        self.start = time.time()

    def train_one_epoch(self, ep):
        pass

    def test_one_epoch(self, ep):
        pass

    def forward(self):
        if self.mode == "Train":
            for ep in range(self.total_epoch-self.last_epoch):
                ep = ep+self.last_epoch
                self.train_one_epoch(ep)
                self.test_one_epoch(ep)
        else:
            self.test_one_epoch()


class seq_solver(solver):

    def __init__(self, criterion, model, lmdb, optimizer, scheduler, train_solver, others, pre_train = False):
        super(seq_solver, self).__init__(criterion, model, lmdb, optimizer, scheduler, train_solver, others, pre_train)

        last_epoch = -1
        self.last_epoch = 0
        if model['loading_epoch'] != 0:
            check_point = torch.load(model['model_path'].format(model['loading_epoch']))
            last_epoch = model['loading_epoch']
            self.last_epoch = last_epoch + 1
            self.model.load_state_dict(check_point['state_dict'])
            self.optimizer.load_state_dict(check_point['optimizer'])

        self.scheduler = scheduler['scheduler_type'](self.optimizer, 
                        milestones = scheduler['milestones'], gamma = scheduler['gamma'], last_epoch=last_epoch)

    def train_one_epoch(self, ep):
        self.model.train()
        loss_aver = 0
        if self.scheduler is not None:
            self.scheduler.step()
            print('learning_rate: ', self.scheduler.get_lr())

        for it, (sample_batched) in enumerate(self.lmdb_train):
            inputs = sample_batched['image'].squeeze(0).float()
            labels = sample_batched['label'].squeeze(0)
            inputs = inputs.cuda()

            output = self.model(inputs)
            output = output.transpose(0, 1)
            loss = self.loss_layer(output, labels, inputs)

            self.optimizer.zero_grad()
            loss.backward()
            loss = loss.data[0]
            l2_norm = cindy_gradient_clip(self.model)

            if not np.isnan(l2_norm.cpu()):
                self.optimizer.step()
            else:
                print('l2_norm: ', l2_norm)
                l2_norm = 0

            if it == 0:
                loss_aver = loss
            if(loss > 10000 or loss < 0):
                loss = loss_aver
                print('inf')
            if not np.isnan(loss):           
                loss_aver = 0.9*loss_aver+0.1*loss            

            if (ep == 0 and it < 1000 and it % 100 == 0) or (it % 1000 == 0):
                prediction, result = self.loss_layer.result_analysis_recall()
                out = self.print_result(result)
                print('Train: Epoch: %3d it: %6d, loss: %.4f CR: %.4f  AR: %4f, l2_norm: %.4f, recall: %.4f, precision: %.4f, WER: %.4f' % 
                    (ep, it, loss_aver, out[0], out[1], l2_norm, out[2], out[3], out[4]))

        prediction, result = self.loss_layer.result_analysis_recall()
        out = self.print_result(result)
        print('Train: Epoch: %3d it: %6d, loss: %.4f CR: %.4f  AR: %4f, l2_norm: %.4f, recall: %.4f, precision: %.4f, WER: %.4f' % 
            (ep, it, loss_aver, out[0], out[1], l2_norm, out[2], out[3], out[4]))
        f = open(self.log_path+'TrainAccuracyLoss.log', 'a')
        f.write('%d %f %f %f %f %f %f\n' % (ep, loss_aver, out[0], out[1], out[2], out[3], out[4]))
        f.close()

        torch.save({
            'epoch': ep,
            'state_dict': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            }, self.model_path.format(ep))  


    def test_one_epoch(self, ep=0):
        self.model.eval()
        loss_aver = 0
        total_result = [0]*9
        for it, (sample_batched) in enumerate(self.lmdb_test):

            inputs = sample_batched['image'].squeeze(0).float()
            labels = sample_batched['label'].squeeze(0)

            inputs = inputs.cuda()
            output = self.model(inputs)
            output = output.transpose(0, 1)
            loss = self.loss_layer(output, labels, inputs)
            prediction, result = self.loss_layer.result_analysis_recall()
            loss = loss.data[0]
            if it == 0:
                loss_aver = loss
            loss_aver = 0.9*loss_aver+0.1*loss      

            for i, ele in enumerate(result):
                total_result[i] += ele 

            if it % 5000 == 0:
                out = self.print_result(total_result)
                print('Test :  Epoch: %3d it: %6d, loss: %.4f CR: %.4f  AR: %4f  delete: %4f, replace: %4f, insert: %4f, len : %4d, recall: %.4f, precision: %.4f, WER: %.4f, right_word: %4d, total word: %.4f' % 
                    (ep, it, loss_aver, out[0], out[1], total_result[0], total_result[1], total_result[2], total_result[3], out[2], out[3], out[4], total_result[7], total_result[8]))

        out = self.print_result(total_result)
        f = open(self.log_path+'TestAccuracyLoss.log', 'a')
        f.write('%d %f %f %f %f %f %f\n' % (ep, loss_aver, out[0], out[1], out[2], out[3], out[4]))
        f.close()

    def print_result(self, result):
        CR = 1-(float)(result[0]+result[1])/result[3]
        AR = 1-(float)(result[0]+result[1]+result[2])/result[3] 
        recall = float(result[4]) / result[5]
        precision = float(result[4]) / (result[6]+0.000001)                     
        WER = float(result[7])/result[8]
        return [CR, AR, recall, precision, WER]