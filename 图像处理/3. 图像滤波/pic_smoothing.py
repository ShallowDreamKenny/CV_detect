#!/usr/bin/env python3
# coding:utf-8
"""
# File       : pic_smoothing.py
# Time       ：10/13/22 3:33 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("pic1.png")
plt.subplot(321),plt.imshow(img)

# 平均卷积 均值滤波
blur = cv2.blur(img,(3,3))
plt.subplot(322),plt.imshow(blur)

# 方框滤波
box = cv2.boxFilter(img,-1,(3,3),normalize=True)
plt.subplot(323),plt.imshow(box)

#不做归一化会出现超出最大预知的情况
box_non_normalize = cv2.boxFilter(img,-1,(3,3),normalize=False)
plt.subplot(324),plt.imshow(box_non_normalize)

# 高斯滤波
# 更重视卷积核后中间的像素点，即中间的像素点在归一化时权值更高
Gaussian = cv2.GaussianBlur(img,(5,5),1)
plt.subplot(325),plt.imshow(Gaussian),plt.title("gaussi")

# 中值滤波
# 中间值作为滤波后的结果
median = cv2.medianBlur(img,5)
plt.subplot(326),plt.imshow(median),plt.title("median")

# hstack可以把图像拼成一行
# vstack可以吧图像拼成一列
res = np.hstack((img,blur,box_non_normalize,Gaussian,median))
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.show()