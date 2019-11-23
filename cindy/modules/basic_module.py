
import torch.nn as nn
import torch.nn.functional as F

class ResidualLSTM(nn.Module):
    def __init__(self, drop_out = 0.0, layer_num = 3, rnn_size = 512):
        super(ResidualLSTM, self).__init__()

        self.lstm_list = nn.ModuleList()
        self.drop_list = nn.ModuleList()
        self.layer_num = layer_num
        for layer in range(self.layer_num):
            self.lstm_list.append(nn.LSTM(rnn_size, rnn_size, num_layers=1))
            self.drop_list.append(nn.Dropout(p=drop_out))


    def forward(self, input):
        for layer in range(self.layer_num):
            output,_ = self.lstm_list[layer](input)
            output = output+input
            input = self.drop_list[layer](output)

        return input


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.3)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  
        output = output.view(T, b, -1)

        return output