import os
import cv2, json
import os.path as osp
import numpy as np
import random

def func1():
    file_info = {}

    base_dir = "/media/maweihong/Weihong_2T/拷贝/古籍数据/古籍版本_20191010/Textline_train/"
    img_dirs = ["MTH1200/img",
                "TKH/img"]
    for dir_ in img_dirs:
        for file in os.listdir(osp.join(base_dir, dir_)):
            img = cv2.imread(osp.join(base_dir, dir_, file))
            h, w, _ = img.shape
            file_info[file] = [h,w]

    with open("img_info.json", "w") as f:
        data = json.dump(file_info, f)


def func2():
    chars = []
    base_dir = "/media/maweihong/Weihong_2T/拷贝/古籍数据/古籍版本_20191010/Textline_train/"
    txt_files = ["MTH1200/label.txt",
                "TKH/label.txt"]
    for file_ in txt_files:
        data = open(osp.join(base_dir, file_), 'r').read().splitlines()
        for line in data:
            val = line.split(',')[1]
            for char in val:
                if char not in chars and char!='#':
                    chars.append(char)
    f = open('../../data/dict.txt', 'w')
    for char in chars:
        f.write(char+'\n')
    f.close()
    return

def func3():
    data = open("../../data/Textline_train/MTH1200/label.txt", 'r').read().splitlines()
    test = random.sample(data, 1000)
    f = open("test_label.txt", "w")
    for val in test:
        f.write(val+"\n")
    f.close()
    train = list(set(data).difference(test))
    f = open("train_label.txt", "w")
    for val in train:
        f.write(val+"\n")
    f.close()
    return

if __name__ == "__main__":
    func3()