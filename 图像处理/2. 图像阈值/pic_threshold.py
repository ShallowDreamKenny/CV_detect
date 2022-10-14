#!/usr/bin/env python3
# coding:utf-8
"""
# File       : pic_threshold.py
# Time       ：10/13/22 3:15 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2
from matplotlib import pyplot as plt

# threshold 只能处理单通道图像，通常来说使灰度图
# 5中转换方法，
#   BINARY 超过阈值部分取maxval，小于取0
#   BINARY_INV 翻转BINARY
#   TRUNC   大于阈值部分设为阈值，否则不变
#   TOZERO  大于阈值不变，小于取0
#   TOZERO_INV  翻转TOZERO

cat = cv2.imread("../1. 基本操作/cat.jpg")

cat_gray = cv2.cvtColor(cat,cv2.COLOR_BGR2GRAY)
cat = cv2.cvtColor(cat,cv2.COLOR_BGR2RGB)
ret, thresh1 = cv2.threshold(cat_gray,127,255,cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(cat_gray,127,255,cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(cat_gray,200,255,cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(cat_gray,0,255,cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(cat_gray,127,255,cv2.THRESH_TOZERO_INV)
title = ['ORIGINAL',"B",'B_I','TRUNC','TO','YO_INV']
images = [cat,thresh1,thresh2,thresh3,thresh4,thresh5]

for i in range(6):
    plt.subplot(2,3,i+1), plt.imshow(images[i],'gray')
    plt.title(title[i])

plt.show()