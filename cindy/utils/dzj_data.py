
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

AUG = settings.data_args['AUG']
DropLabel = settings.data_args['DropLabel']
DropRatio = settings.data_args['DropRatio']
USE_SYNDATA = settings.data_args['USE_SYNDATA']
SYN_RATIO = settings.data_args['SYN_RATIO']
Train_Batch = settings.data_args['Train_Batch']
HEIGHT = settings.data_args['HEIGHT']


class DZJDataset(Dataset):

    def __init__(self):
        self.dict = {}
        data = open('data/dict.txt', 'r').read().splitlines()
        self.alphabet = [line for line in data]
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i

        self.imgs_num = 0
        self.section = [1, 3, 6, 9, 12, 15]

        self.image_stages = [[] for i in range(len(self.section)+1)]
        self.label_stages = [[] for i in range(len(self.section)+1)]
        self.fixed_h = HEIGHT
        self.batch_size = Train_Batch

        UNIT_WIDTH = HEIGHT*2

        base_dir = "data/Textline_train"
        dir_name = ["TKH", "MTH1200"]
        for dir_ in dir_name:
            file_path = osp.join(base_dir, dir_, 'train_label.txt')
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
                index = bisect.bisect(self.section, int(res_w/UNIT_WIDTH))
                label = [self.dict[char] for char in elements if char!='#' ]

                self.image_stages[index].append(res_img)
                self.label_stages[index].append(label)
                self.imgs_num += 1

    def __getitem__(self, idx):
        searching = True
        while searching:
            stage = random.randint(0,len(self.section))
            if len(self.image_stages[stage])>=self.batch_size:
                searching = False
        images = [];    labels = []
        for i in range(self.batch_size):
            image, label = self.query(stage)
            images.append(image)
            labels.append(label)

        padded_images = self.to_image_padded(images)

        # TODO: image process(augmentation, normalization)
        if AUG:
            seq = iaa.Sequential([
                    iaa.Sometimes(0.5,
                        iaa.GaussianBlur(sigma=(0, 0.5))
                    ),
                    iaa.ElasticTransformation(alpha=(0, 2.5), sigma=0.5),
                    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                    ], random_order=True)
            padded_images = seq(images=padded_images[:,0,:,:])
            padded_images = padded_images[:,np.newaxis,:,:]

        # DEBUG:
        # cv2.imwrite("4.jpg", padded_images[0])
        # Normalization
        padded_images = (padded_images - 127.5)/127.5
        sample = {}
        sample['image'] = padded_images
        sample['label'] = np.array(labels)
        return sample

    def __len__(self):
        return self.imgs_num

    def query(self, stage):
        if USE_SYNDATA:
            pass
        else:
            idx = random.randint(0, len(self.image_stages[stage])-1)
            image = self.image_stages[stage][idx]
            label = self.label_stages[stage][idx]

        if DropLabel:
            drop_list = []
            for i in range(len(label)):
                if random.random() < DropRatio:
                    drop_list.append(i)
            if len(drop_list) == len(label):
                drop_list.pop(np.random.randint(len(drop_list)))
            label = [label[i] for i in range(len(label)) if i not in drop_list]

        binary_img = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)[1]
        text = np.zeros(70)-1
        text[:len(label)] = label
        text = text.astype('int')
        return binary_img, text

    def to_image_padded(self, images):
        batch_shape = tuple(max(s) for s in zip(*[img.shape for img in images]))
        padded_images = np.ones((self.batch_size, 1, batch_shape[0], batch_shape[1]))*255
        for index, image in enumerate(images):
            h, w = image.shape
            diff_w = batch_shape[1]-w
            if diff_w>0:
                start = np.random.randint(diff_w)
                padded_images[index, 0, :batch_shape[0], start:(start+w)] = image
            else:
                padded_images[index, 0, :batch_shape[0], :] = image
        return padded_images