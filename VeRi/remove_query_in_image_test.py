#!/usr/bin/env python
import os, glob, sys

img_path = './image_test_wo_query/*.jpg'
txt_path = './name_query.txt'

with open(txt_path) as f:
    content = f.readlines()
content = [x.strip() for x in content] 

#print content

cnt = 1
for img in sorted(glob.glob(img_path)):

    fname = os.path.basename(img)

    if fname in content:
      print (img, cnt)
      cnt += 1
      os.remove(img)
