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

np.set_printoptions(precision=2,suppress=True,threshold=10000)



class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings=128):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size,bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size+num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_embeddings = num_embeddings
        self.processed_batches = 0
        # self.i2h_bn = nn.BatchNorm1d(hidden_size)
        # self.h2h_bn = nn.BatchNorm1d(hidden_size)

    def forward(self, prev_hidden, feats, cur_embeddings):
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size

        # feats_proj = self.i2h_bn(self.i2h(feats.view(-1,nC)))
        feats_proj = self.i2h(feats.view(-1,nC))
        # prev_hidden_proj = self.h2h_bn(self.h2h(prev_hidden).view(1,nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(-1, hidden_size))
        prev_hidden_proj = self.h2h(prev_hidden).view(1,nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(-1, hidden_size)
        emition = self.score(F.tanh(feats_proj + prev_hidden_proj).view(-1, hidden_size)).view(nT,nB).transpose(0,1)
        self.processed_batches = self.processed_batches + 1
        alpha = F.softmax(emition) 
        
        context = (feats * alpha.transpose(0,1).contiguous().view(nT,nB,1).expand(nT, nB, nC)).sum(0).squeeze(0) 
        if len(context.size()) == 1:
            context = context.unsqueeze(0)
        context = torch.cat([context, cur_embeddings], 1)
        cur_hidden = self.rnn(context, prev_hidden)
        return cur_hidden, alpha

class BaiAttention(Sequence):
    def __init__(self, input_size, hidden_size, num_classes, num_embeddings=128):
        super(BaiAttention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_embeddings)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.generator = nn.Linear(hidden_size, num_classes+1)
        self.char_embeddings = nn.Parameter(torch.randn(num_classes+2, num_embeddings))
        self.num_embeddings = num_embeddings
        self.processed_batches = 0
        self.num_classes = num_classes+1
        self.EOS_token = num_classes+1-1
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index = -1)
        self.image_count = 0

    def forward(self, feats, label):

        # text_length, text
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)        
        batch_size, total_len = label.size()
        text_length = torch.zeros(nB)

        self.label = label.clone()
        label_train = label.clone()

        for bn in range(batch_size):
            for tn in range(total_len):
                text_length[bn] += 1;
                if label[bn][tn] == -1:
                    label_train[bn][tn] = self.EOS_token
                    break;
        num_steps = int(text_length.data.max().item())
        label_train = label_train[:,:num_steps]
        target = label_train.clone()+1
        target = torch.cat((torch.zeros(target.size(0),1).double(),target.double()),dim=-1);


        # pdb.set_trace()

        if self.training and np.random.rand() < 0.5:
            self.processed_batches = self.processed_batches + 1
            hidden_size = self.hidden_size
            input_size = self.input_size
            assert(input_size == nC)
            assert(nB == text_length.numel())

            targets = Variable(target.long().cuda().transpose(0,1).contiguous())
            output_hiddens = Variable(torch.zeros(num_steps, nB, hidden_size).type_as(feats.data))
            hidden = Variable(torch.zeros(nB,hidden_size).type_as(feats.data))
            max_locs = torch.zeros(num_steps, nB)
            max_vals = torch.zeros(num_steps, nB)
            for i in range(num_steps):
                cur_embeddings = self.char_embeddings.index_select(0, targets[i])
                hidden, alpha = self.attention_cell(hidden, feats, cur_embeddings)
                output_hiddens[i] = hidden
                if self.processed_batches % 500 == 0:
                    max_val, max_loc = alpha.data.max(1)
                    max_locs[i] = max_loc.cpu()
                    max_vals[i] = max_val.cpu()
            if self.processed_batches % 500 == 0:
                print('max_locs:',['%.1f,' % i for i in max_locs[:,0].data.numpy().tolist()])
                print('max_val :',['%.3f,' % i for i in max_vals[:,0].data.numpy().tolist()])
            probs = self.generator(output_hiddens) # unsolved dim 1
            preds = probs.contiguous().view(-1,self.num_classes);
            labels = label_train.transpose(0,1).contiguous().view(label_train.numel()).long().cuda()
            loss = self.criterion(preds, labels)
            self.softmax = probs
            return loss
        else:
            self.processed_batches = self.processed_batches + 1

            hidden_size = self.hidden_size
            input_size = self.input_size
            assert(input_size == nC)
            
            num_steps = int(text_length.data.max().item())
            # num_labels = text_length.data.sum()

            hidden = Variable(torch.zeros(nB,hidden_size).type_as(feats.data))

            targets_temp = Variable(torch.zeros(nB).long().cuda().contiguous())
            probs = Variable(torch.zeros(nB*num_steps, self.num_classes).cuda())
            for i in range(num_steps):
                cur_embeddings = self.char_embeddings.index_select(0, targets_temp)
                hidden, alpha = self.attention_cell(hidden, feats, cur_embeddings)
                hidden2class = self.generator(hidden)
                probs[i*nB:(i+1)*nB] = hidden2class
                _, targets_temp = hidden2class.max(1)
                targets_temp += 1
            self.softmax = probs.view(num_steps, nB, self.num_classes)
            probs = probs.view(-1, self.num_classes).contiguous()
            labels = label_train.transpose(0,1).contiguous().view(label_train.numel()).long().cuda()
            loss = self.criterion(probs, labels)
            return loss

    def result_analysis_recall(self):

        result = self.softmax.data.topk(1)[1]
        result = result.squeeze();
        result = result.type(torch.LongTensor)

        batch_size,_ = self.label.size()
        label_batch = torch.transpose(self.label,0,1)
        # label_batch = label_batch.data

        delete_total = 0
        replace_total = 0
        insert_total = 0
        len_total = 0
        word_total = 0
        all_total = 0
        pre_total = 0
        correct_count = 0
        # print(self.processed_batches)
        # if self.processed_batches > 5000:
        show_result_pred = ''
        show_result_label = ''
        for i in range(batch_size):
            pre_list = []
            # pdb.set_trace()
            for ln in result[:,i]:
                ln = ln.item()
                if ln != self.EOS_token:
                    pre_list.append(ln);
                else:
                    break;
            label_list = []
            for ln in label_batch[:,i]:
                if ln != -1:
                    label_list.append(int(ln));
                else:
                    break;
            # if i == 0:
                # print('label_list:', label_list)
                # print('pre_list:', pre_list)            

            # if (self.training and i == 0) or np.random.rand() < 0.0001:
            #     alphabet = '_$!$#$"$\'$&$)$($+$*$-$,$/$.$1$0$3$2$5$4$7$6$9$8$;$:$?$A$C$B$E$D$G$F$I$H$K$J$M$L$O$N$Q$P$S$R$U$T$W$V$Y$X$Z$a$c$b$e$d$g$f$i$h$k$j$m$l$o$n$q$p$s$r$u$t$w$v$y$x$z$|'.split('$')
            #     print(''.join([alphabet[i+1] for i in label_list]))
            #     print(''.join([alphabet[i+1] for i in pre_list]))

            # if self.training and i <= debug[1]:
            # if i < self.debug[0]:
            if i == 0:
                alphabet = '_$!$#$"$\'$&$)$($+$*$-$,$/$.$1$0$3$2$5$4$7$6$9$8$;$:$?$A$C$B$E$D$G$F$I$H$K$J$M$L$O$N$Q$P$S$R$U$T$W$V$Y$X$Z$a$c$b$e$d$g$f$i$h$k$j$m$l$o$n$q$p$s$r$u$t$w$v$y$x$z$|'.split('$')
                show_result_pred = show_result_pred + '  ' + ''.join([alphabet[i+1] for i in pre_list])
                show_result_label = show_result_label + '  ' + ''.join([alphabet[i+1] for i in label_list])

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
                       
        if self.training or np.random.rand() < 0.01:
            # print('')
            # print(self.image_count)
            print('label:', show_result_label)
            print('pred: ',show_result_pred)
            #print(np.array(self.debug[2]))

            # from PIL import Image, ImageDraw
            # im = Image.fromarray(self.debug[1][0,0].cpu().data.numpy()*127.5+127.5)
            # draw = ImageDraw.Draw(im)
            # for pos in self.debug[2]:
            #     x = (pos-1)*16+16
            #     draw.ellipse((x-8,160-8, x+8, 160+8), fill = 128)                
            # im = im.convert('RGB')
            # if self.training:
            #     im.save('/media/pci/256c7be3-9106-49d7-a195-2dbc53b60824/zecheng/output/tmp_output001/image/00train_%05d.jpg'%  self.image_count)
            # else:
            #     im.save('/media/pci/256c7be3-9106-49d7-a195-2dbc53b60824/zecheng/output/tmp_output001/image/01test_%05d.jpg'%  self.image_count)

            # self.image_count += 1            


        rec_result = [delete_total, replace_total, insert_total, len_total, correct_count, len_total, pre_total, word_total, all_total]
        return result, rec_result