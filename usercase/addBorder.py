# -*- coding: utf-8 -*-
import glob
import cv2
import os



search_path = "D:\\pythonworkspace\\tf_unet\\tf_unet\\demo\\IRholder\\ImageResize\\*.png"

files = glob.glob(search_path)

save_path = 'borderImage'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

for i in files:
    imageName = i.split('\\')[-1]
    img = cv2.imread(i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reflect = cv2.copyMakeBorder(gray,50,50,50,50,cv2.BORDER_REFLECT)
    dstPath = os.path.join(save_path,i)
    cv2.imwrite(i,reflect)
