#!/usr/bin/env python3
# coding:utf-8
"""
# File       : pic_Gradient.py
# Time       ：10/13/22 4:35 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("pic.png")
plt.subplot(131),plt.imshow(img)
erode = cv2.erode(img,(5,5),iterations=8)
plt.subplot(132),plt.imshow(erode)
dilate = cv2.dilate(img,(5,5),iterations=8)
plt.subplot(133),plt.imshow(dilate)

# 梯度运算 = 膨胀 - 腐蚀
gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,(5,5))
cv2.imshow("gradient",gradient)
cv2.waitKey(0)

plt.show()
