#!/usr/bin/env python
import os, glob, sys

img_path = './image_train/*.jpg'
txt_path = './label_text.txt'

txt_f = open(txt_path, 'w')

prev_cls = '0001'
new_label = 0

for img in sorted(glob.glob(img_path)):

    fname = os.path.basename(img)
    cls = fname[:4]

    if (cls != prev_cls):
        print (fname, cls, new_label)
        # label text: cls_id, real vehicle id
        txt_f.write(str(new_label) + ' ' + prev_cls + '\n')
        new_label += 1


    prev_cls = cls


# for last class
txt_f.write(str(new_label) + ' ' + prev_cls + '\n')

txt_f.close()
