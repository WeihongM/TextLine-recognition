from cindy.modules.model import *
import torch
import numpy as np
import itertools
import cv2

def decode(input_):
    length,batch_size,dim = input_.shape
    out_best = torch.max(input_, 2)[1].data
    out_best = out_best.cpu().numpy()
    out_best = out_best.reshape(-1,batch_size)
    out_best = out_best.transpose()
    out_best_list = [0]*batch_size
    for j in range(out_best.shape[0]):
        out_best_new = [k for k, g in itertools.groupby(out_best[j])]
        out_best_list[j] = [int(x) for x in out_best_new if x != 0]

    return out_best_list

def img_process(img_path):
    fixed_h = 96
    img = cv2.imread(img_path, 0)
    img = cv2.flip(img, 1)
    img = cv2.transpose(img)
    h, w = img.shape
    res_h, res_w = fixed_h, int(w*fixed_h/h)
    res_img = cv2.resize(img, (res_w, res_h), interpolation = cv2.INTER_AREA)
    binary_img = cv2.threshold(res_img, 130, 255, cv2.THRESH_BINARY)[1]
    binary_img = (binary_img - 127.5)/127.5
    return binary_img

char_dict = {}
data = open('data/dict.txt', 'r').read().splitlines()
alphabet = [line for line in data]
for i, item in enumerate(alphabet):
    char_dict[i+1] = item

img_path = "data/Textline_train/TKH/img/0001_001_26_01_0.jpg"
img = img_process(img_path)
binary_img = torch.from_numpy(img[np.newaxis, np.newaxis, :, :]).float()

model = ChineseEncoderDecoder()
checkpoint = torch.load("model/dzj-04.pkl")
pretrained_dict = checkpoint['state_dict']
pretrained_dict = {k.replace("module.", ''): v for k, v in pretrained_dict.items()}
model.load_state_dict(pretrained_dict)

output = model(binary_img)
output = output.transpose(0, 1)
out_best_list = decode(output)
print([char_dict[val] for val in out_best_list[0]])