
import time, math, os, cv2, random, bisect
import numpy as np
import os.path as osp
import imgaug as ia
import imgaug.augmenters as iaa

import torch
from torch.utils.data import Dataset
import settings

torch.manual_seed(1671)
torch.cuda.manual_seed_all(1671)

HEIGHT = settings.data_args['HEIGHT']

class DZJDatasetTest(Dataset):

    def __init__(self):
        self.dict = {}
        data = open('data/dict.txt', 'r').read().splitlines()
        self.alphabet = [line for line in data]
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i

        self.imgs_num = 0
        self.images = [];       self.labels = []

        self.fixed_h = HEIGHT
        self.batch_size = 1

        UNIT_WIDTH = HEIGHT*2

        base_dir = "data/Textline_train"
        dir_name = ["TKH", "MTH1200"]
        for dir_ in dir_name:
            file_path = osp.join(base_dir, dir_, 'test_label.txt')
            data = open(file_path, 'r').read().splitlines()
            for index, line in enumerate(data):
                img_filename = line.split(',')[0]
                elements = line.split(',')[1]
                img = cv2.imread(osp.join(base_dir, dir_, 'img', img_filename), 0)
                img = cv2.flip(img, 1)
                img = cv2.transpose(img)
                h, w = img.shape
                res_h, res_w = self.fixed_h, int(w*self.fixed_h/h)
                res_img = cv2.resize(img, (res_w, res_h), interpolation = cv2.INTER_AREA)
                label = [self.dict[char] for char in elements if char!='#' ]

                self.images.append(res_img)
                self.labels.append(label)
                self.imgs_num += 1

    def __getitem__(self, idx):

        image = self.images[idx]
        label = self.labels[idx]

        binary_img = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)[1]
        text = np.zeros(70)-1
        text[:len(label)] = label
        text = text.astype('int')

        # Normalization
        image = (binary_img - 127.5)/127.5
        sample = {}
        image = image[np.newaxis, np.newaxis, :, :]
        sample['image'] = image
        sample['label'] = np.array(text).reshape(1,70)
        return sample

    def __len__(self):
        return self.imgs_num