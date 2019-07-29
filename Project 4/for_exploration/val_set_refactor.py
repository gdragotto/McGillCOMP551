import io
import pandas as pd
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir

#from https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/utils/tiny-imgnet-val-reformat.ipynb
target_folder = '/home/yang94/project/yang94/courses/COMP551/miniproject4/tiny-imagenet-200/val/'

val_dict = {}
with open(target_folder + 'val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]

paths = glob.glob(target_folder + 'images/*')
paths[0].split('/')[-1]
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))

for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    dest = target_folder + str(folder) + '/' + str(file)
    move(path, dest)

os.remove('/home/yang94/project/yang94/courses/COMP551/miniproject4/tiny-imagenet-200/val/val_annotations.txt')
rmdir('/home/yang94/project/yang94/courses/COMP551/miniproject4/tiny-imagenet-200/val/images')
