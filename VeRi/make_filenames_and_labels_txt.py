#!/usr/bin/env python
import os, glob, sys

img_path = './image_train/*.jpg'
filenames_txt_path = './filenames_image_train.txt'
labels_txt_path = './labels_image_train.txt'

filenames_txt_f = open(filenames_txt_path, 'w')
labels_txt_f = open(labels_txt_path, 'w')

prev_cls = '0001'
new_label = 0

for img in sorted(glob.glob(img_path)):

    fname = os.path.basename(img)
    cls = fname[:4]

    if (cls != prev_cls):
        new_label += 1

    print (fname, cls, new_label)
    filenames_txt_f.write(fname + '\n')
    labels_txt_f.write(str(new_label) + '\n')

    prev_cls = cls


filenames_txt_f.close()
labels_txt_f.close()

