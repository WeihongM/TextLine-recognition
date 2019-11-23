# -*- coding: utf-8 -*-
import numpy as np
from torch.autograd import Variable
from cindy.utils.basic import cal_distance
import torch
import pdb
import itertools
from warpctc_pytorch import CTCLoss
import torch.nn as nn
import torch.nn.functional as F
import random
from PIL import Image, ImageDraw
import math
import cv2

import numpy as np  
np.set_printoptions(threshold =1000000, suppress=True, precision=2) 

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.count_n = 0

    def loss(self, label, output):
        pass;

    def decode_batch(self):
        # pdb.set_trace()
        length,batch_size,dim = self.softmax.shape
        out_best = torch.max(self.softmax, 2)[1].data
        out_best = out_best.cpu().numpy()
        out_best = out_best.reshape(-1,batch_size)
        out_best = out_best.transpose()
        self.tmp_result = out_best
        out_best_list = [0]*batch_size
        for j in range(out_best.shape[0]):
            out_best_new = [k for k, g in itertools.groupby(out_best[j])]
            # out_best_new = out_best[j]
            out_best_list[j] = [int(x) for x in out_best_new if x != 0]
        return out_best_list

    def decode_batch_whang(self):
        # pdb.set_trace()
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
        return out_best_list

    def decode_batch_LinkACE(self):
        # pdb.set_trace()
        link = self.link.data.cpu().numpy()
        length,batch_size,dim = self.softmax.shape
        out_best = torch.max(self.softmax, 2)[1].data
        out_best = out_best.cpu().numpy()
        out_best = out_best.reshape(-1,batch_size)
        out_best = out_best.transpose()
        self.tmp_result = out_best
        out_best_list = [0]*batch_size
        for j in range(out_best.shape[0]):
            out_best_adjust = [];
            for id, vv in enumerate(out_best[j][:-1]):
                out_best_adjust.append(vv)
                if vv == out_best[j][id+1] and vv != 0 and (self.link[j,0,0,id+1] < 0.5 or self.link[j,1,0,id] < 0.5):
                    out_best_adjust.append(0)
            out_best_adjust.append(out_best[j][-1])
            out_best_new = [k for k, g in itertools.groupby(out_best_adjust)]
            # out_best_new = out_best[j]
            out_best_list[j] = [int(x) for x in out_best_new if x != 0]
        return out_best_list


    def result_analysis(self):
        prediction = self.decode_batch()

        batch_size = self.softmax.size(1)
        delete_total = 0
        replace_total = 0
        insert_total = 0
        len_total = 0
        for i in range(batch_size):
            pre_list = prediction[i]
            label_list = self.label[i][self.label[i]!=-1].tolist()
            label_list = [int(ele)+1 for ele in label_list]
            # if i == 0:
            #     print('label_list:', label_list)
            #     print('pre_list:', pre_list)              
            distance, (delete, replace, insert) = cal_distance(label_list, pre_list)
            delete_total += delete
            replace_total += replace
            insert_total += insert
            len_total += len(label_list)
        result = [delete_total, replace_total, insert_total, len_total]
        return prediction, result  


    def result_analysis_recall(self):
        prediction = self.decode_batch()
        batch_size = self.softmax.size(1)
        delete_total = 0
        replace_total = 0
        insert_total = 0
        len_total = 0
        correct_count = 0
        pre_total = 0
        word_total = 0
        all_total = 0

        for i in range(batch_size):
            pre_list = prediction[i]
            label_list = self.label[i][self.label[i]!=-1].tolist()
            label_list = [int(ele)+1 for ele in label_list]

            distance, (delete, replace, insert) = cal_distance(label_list, pre_list)
            delete_total += delete
            replace_total += replace
            insert_total += insert
            len_total += len(label_list)
            pre_total += len(pre_list)
            correct_count += len(label_list) - delete - replace;

            if distance == 0:
                word_total += 1 
            all_total += 1
        # print (correct_count , len_total, pre_total)
        # recall = float(correct_count) / len_total
        # precision = correct_count / (pre_total+0.000001)
        result = [delete_total, replace_total, insert_total, len_total ,correct_count, len_total, pre_total, word_total, all_total]
        return prediction, result 


    def show_error_example(self,index):
        pre_list = self.tmp_result[index]
        label_list = self.label[index][self.label[index]!=-1].tolist()
        raw_pred = ''
        for j in pre_list:
            a = '%2s' % self.dict[int(j)]
            raw_pred += a
        print(raw_pred)
        raw_pred = ''
        for j in label_list:
            a = '%2s' % self.dict[int(j)+1]
            raw_pred += a
        print(raw_pred)    


class OrangeACELINKNEW(Sequence):
    def __init__(self, class_num, para = 1.0):
        super(OrangeACELINKNEW, self).__init__()
        self.softmax = None;
        self.label = None;
        self.class_num = class_num
        self.para = para
        self.dict='_0123456789abcdefghijklmnopqrstuvwxyz'

        self.log_num = 0 # which picture logging
        # -np.sum(np.log(ll)*ln)+np.sum(np.log(ln)*ln) = 0.0019442696075796517

        self.acutual_num = 8*24

        ll = np.array((self.acutual_num-7*4, 1*4, 2*4, 3*4, 1.0*4))/self.acutual_num
        ln = np.array((self.acutual_num-6*4, 1*4, 2*4, 2*4, 1.0*4))/self.acutual_num
        
        self.coeff = -np.sum(np.log(ll)*ln)+np.sum(np.log(ln)*ln)
        print('coeff: ', self.coeff)

    def draw(self, fea_map_softmax, sample_range):
        """ Visualize the 2D feature map."""
        batch_size, class_num, height, width = fea_map_softmax.shape
        fea_map_softmax = fea_map_softmax.permute(0,2,3,1)
        fea_map_maxindex = fea_map_softmax.topk(1)[1]    
        for sn in range(sample_range):
            max_map = fea_map_maxindex[sn,:,:,0]
            for i in range(height):
                raw_pred=''
                for j in range(width):
                    # a = '%2s' % self.dict[tmp[i*width+j].data.cpu().numpy().tolist()]
                    a = '%2s' % self.dict[max_map[i][j].item()]
                    raw_pred += a
                print(raw_pred) 

    def draw2D(self, fea_map_softmax, image, link):
        """ Visualize the 2D feature map for ACE or LinkACE.

        Arguments:
            fea_map_softmax (Variable): softmax output. shape of (batch_size, class_num, height, width)
        """

        if self.training: # Training phase
            if self.log_out:
                print('\n######################################################################')
                print('Pic: %d' % self.log_num)
                self.draw(fea_map_softmax, 1)
                # im = Image.fromarray(image[0,0].cpu().numpy())
                # im = im.convert('RGB')
                # im.save('/media/pci/256c7be3-9106-49d7-a195-2dbc53b60824/zecheng/output/tmp_output001/image/%05d_train.jpg'% self.log_num)
                self.log_num +=1
                print("torch.sum(link[0][0]): ", torch.sum(link[0][0]))
                print("torch.mean(link[0][0]): ", torch.mean(link[0][0]))
                print(link[0][0].data.cpu().numpy())            
                print(link[0][1].data.cpu().numpy())            
                # print(link[0][2].data.cpu().numpy())            
                # print(link[0][3].data.cpu().numpy())            

        # else: # Testing Phase
        #     self.draw(fea_map_softmax, 1)


    def forward(self, result, label):


        """ Calculate LinkACE.

        Arguments:
            result (tuple): (image, input, link). 
                image: original input. shape of (batch_size, 1, 100, 100)
                input: softmax output. shape of (batch_size, calss_num+1 (27), height, width)
                link : link output. shape of (batch_size, 4, height, width)
        """
        if random.random() < 0.001:
            self.log_out = True
        else:
            self.log_out = False

        image, input, link = result
        self.link = link
        bs, dim, height, width = input.size()

        assert self.acutual_num == height*width, 'self.acutual_num error'

        num = height*width # num can be smaller that height*width
        ori_num = height*width # ori_num must be height*width

        self.draw2D(input, image, link) 

        input = input.view(bs, dim, -1)
        self.softmax = input.permute(2,0,1);
        

        # ************************************************************************************************
        # Link ACE | 2D-CTC-like 
        # @orange
        # ************************************************************************************************

        # get init result of ACE
        sum_result = torch.sum(input,2) 

        # get link result
        input = input.view(bs, dim, height, width)

        # if and only if the right-side link of the i-th line and the left-side link of the (i+1)-th line are big enough
        left_right = torch.cat((link[:,1:2,:,:-1],link[:,0:1,:,1:]), dim = 1)
        lr_min = torch.min(left_right, dim = 1, keepdim = True)[0]
        # if and only if the down-side link of the j-th row and the up-side link of the (j+1)-th line are big enough
        up_down = torch.cat((link[:,3:4,:-1,:],link[:,2:3,1:,:]), dim = 1)
        ud_min = torch.min(up_down, dim = 1, keepdim = True)[0]


        bb, dd, hh, ww = input.size()
        input = input.view(bs, dim, 1, height, width)
        # if and only if the prediction of the same character at the i-th and (i+1)-th line are big enough
        lr_v = torch.cat((input[:,:,:,:,:-1],input[:,:,:,:,1:]), dim = 2)
        lr_v_min = torch.min(lr_v, dim = 2, keepdim = False)[0]
        # if and only if the prediction of the same character at the j-th and (j+1)-th row are big enough
        ud_v = torch.cat((input[:,:,:,:-1,:],input[:,:,:,1:,:]), dim = 2)
        ud_v_min = torch.min(ud_v, dim = 2, keepdim = False)[0]

        # if and only if the prediction of the same character of adjacent lines or rows, and their link are big enough.
        horizontal = lr_v_min*lr_min
        vertical = ud_v_min*ud_min        

        horizontal = torch.sum(horizontal.view(bs, dim, -1),2)
        vertical = torch.sum(vertical.view(bs, dim, -1),2)

        # this may cause sum_result to be negative, and loss nan.
        # erase the extra non-blank prediction, and append it to the blank prediction.
        otherAll = horizontal[:,1:] + vertical[:,1:]
        sum_result[:,1:] = sum_result[:,1:] -  otherAll
        movement = torch.sum(otherAll,1)
        sum_result[:,0] = sum_result[:,0] + movement




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

        sum_result = sum_result/self.acutual_num
        label_array = label_array/self.acutual_num
        label_var = Variable(torch.from_numpy(label_array).type(torch.cuda.FloatTensor), requires_grad=False).cuda()
        # loss = (-torch.sum(torch.log(sum_result)*label_var))/batch_size + 0.01*torch.sum(link)
        
        loss_ACE =  (-torch.sum(torch.log(sum_result)*label_var))/batch_size # original ACE
        loss_PN = torch.sum(-sum_result[sum_result<0]) # punish negative probability for every class  
        loss_link = torch.sum(link)/batch_size/4/num # link is not very encourage  
        # loss_link_relu = 0               
        # for i in range(4):                                       
        #     loss_link_relu += F.relu(torch.sum(link[:,i,:,:])/batch_size/num-0.2, True); # link is not very encourage
        # loss_link_sum = torch.sum(link)/batch_size/4 # link is not very encourage                                                                  
        # loss_link = (link-0.5).norm(1)/batch_size/4/num # link is not very encourage  
        
        # true_movement = torch.sum(movement)/batch_size/ori_num 
        # blank_loss = torch.relu(true_movement - 0.2)  

        # pdb.set_trace()
        # ll = np.array((93.0, 1, 2, 3, 1))/100
        # ln = np.array((94.0, 1, 2, 2, 1))/100
        # # ln = np.array((20.0, 20, 20, 20, 20))/100
        # np.sum(np.log(ln)*ll)

        # -np.sum(np.log(ll)*ln)+np.sum(np.log(ln)*ln) = 0.0019442696075796517
        loss = loss_ACE + 100*loss_PN + 2*float(self.coeff)*loss_link
        # loss = loss_ACE 

        if self.log_out:
            print('predicted number count for each class:')
            print(sum_result[0].data.cpu().numpy()*num)        
            print('ground truth number count for each class:')
            print(label_var[0].data.cpu().numpy()*num)
            print('ACE loss: ', loss_ACE.item(), 'negative punish loss: ', loss_PN.item(), 'link loss: ', loss_link.item())
            print('*******************************************************************\n')

        return loss