
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import resnet

from .basic_module import ResidualLSTM, BidirectionalLSTM

class ChineseEncoderDecoder(nn.Module):
    def __init__(self):
        super(ChineseEncoderDecoder, self).__init__()
        self.conv1  = nn.Conv2d(1,   16,  kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv2  = nn.Conv2d(16,  64,  kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv3  = nn.Conv2d(64,  128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv4  = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv5  = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv6  = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv7  = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv8  = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv9  = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))        
        self.conv10 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv11 = nn.Conv2d(512, 512, kernel_size=(3, 1), padding=(0, 0), stride=(3, 1))
        self.batch_norm = nn.BatchNorm2d(512)

        self.residual_lstm = ResidualLSTM(drop_out = 0.2, layer_num = 3, rnn_size = 512)
        # self.residual_lstm = ResidualLSTM(drop_out = 0.1, layer_num = 2, rnn_size = 512)
        class_num = 8354
        hidden_num = 512
        self.out = nn.Linear(hidden_num, class_num+1)

    def forward(self, input):
        image = input
        input = F.relu(self.conv1(input), True) # 48 * 1152
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2)) 
        input = F.relu(self.conv2(input), True) # 24 * 576
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))        
        input = F.relu(self.conv3(input), True) # 
        input = F.relu(self.conv4(input), True) # 12 * 288
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))
        input = F.relu(self.conv5(input), True) # 
        input = F.relu(self.conv6(input), True) # 6 * 144
        input = F.relu(self.conv7(input), True) # 
        input = F.relu(self.conv8(input), True) # 
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))
        input = F.relu(self.conv9(input), True) # 3 * 72
        input = F.relu(self.conv10(input), True) # 
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))
        input = F.relu(self.batch_norm(self.conv11(input)), True) # 1 * 72 or 9 * 3

        # print(input.shape)
        # nB, nC, nH, nW = input.size()
        # input = input.view(nB, nC, 1, nH*nW)
        inp = input[:, :, 0, :].transpose(0, 2).transpose(1, 2)
        inp = self.residual_lstm(inp)
        output = self.out(inp)
        output = output.transpose(0, 1)
        return output

class ChineseEncoderDecoder_Large(nn.Module):
    def __init__(self):
        super(ChineseEncoderDecoder_Large, self).__init__()
        self.conv1  = nn.Conv2d(1,   16,  kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv2  = nn.Conv2d(16,  64,  kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv3  = nn.Conv2d(64,  128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv4  = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv5  = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv6  = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv7  = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv8  = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv9  = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))        
        self.conv10 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv11 = nn.Conv2d(512, 512, kernel_size=(3, 1), padding=(0, 0), stride=(3, 1))
        self.batch_norm = nn.BatchNorm2d(512)

        self.residual_lstm = ResidualLSTM(drop_out = 0.2, layer_num = 3, rnn_size = 512)
        # self.residual_lstm = ResidualLSTM(drop_out = 0.1, layer_num = 2, rnn_size = 512)
        class_num = 8354
        hidden_num = 512
        self.out = nn.Linear(hidden_num, class_num+1)

    def forward(self, input):
        image = input
        input = F.relu(self.conv1(input), True) # 48 * 1152
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2)) 
        input = F.relu(self.conv2(input), True) # 24 * 576
        input = F.relu(self.conv3(input), True) # 
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))        
        input = F.relu(self.conv4(input), True) # 12 * 288
        input = F.relu(self.conv5(input), True) # 
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))
        input = F.relu(self.conv6(input), True) # 6 * 144
        input = F.relu(self.conv7(input), True) # 
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))
        input = F.relu(self.conv8(input), True) # 
        input = F.relu(self.conv9(input), True) # 3 * 72
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))
        input = F.relu(self.conv10(input), True) # 
        input = F.relu(self.batch_norm(self.conv11(input)), True) # 1 * 72 or 9 * 3

        nB, nC, nH, nW = input.size()
        input = input.view(nB, nC, 1, nH*nW)            
        inp = input[:, :, 0, :].transpose(0, 2).transpose(1, 2)
        inp = self.residual_lstm(inp)
        output = self.out(inp)
        output = output.transpose(0, 1)
        return output

class ResnetEncoderDecoder(nn.Module):
    def __init__(self):
        super(ResnetEncoderDecoder, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(512)

        resnet34 = models.resnet34(pretrained=True)
        # self.conv1  = nn.Conv2d(1,   64,   kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.cnn = nn.Sequential(*list(resnet34.children())[4:-2])
        self.conv11 = nn.Conv2d(512, 512, kernel_size=(3, 1), padding=(0, 0), stride=(3, 1))
        self.residual_lstm = ResidualLSTM(drop_out = 0.3, layer_num = 3, rnn_size = 512)
        # self.residual_lstm = ResidualLSTM(drop_out = 0.1, layer_num = 2, rnn_size = 512)
        class_num = 8354
        hidden_num = 512
        self.out = nn.Linear(hidden_num, class_num+1)

    def forward(self, input):
        image = input
        # print(input.shape)
        input = F.relu(self.batch_norm1(self.conv1(input)), True)
        input = self.cnn(input)
        # print(input.shape)
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2)) 
        input = F.relu(self.batch_norm2(self.conv11(input)), True)
        # print(input.shape)
        nB, nC, nH, nW = input.size()
        input = input.view(nB, nC, 1, nH*nW)    
        inp = input[:, :, 0, :].transpose(0, 2).transpose(1, 2)
        inp = self.residual_lstm(inp)
        output = self.out(inp)
        output = output.transpose(0, 1)
        return output
