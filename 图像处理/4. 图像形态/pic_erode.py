#!/usr/bin/env python3
# coding:utf-8
"""
# File       : pic_erode.py
# Time       ：10/13/22 3:56 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("pic_erode.jpg")
# plt.subplot(121),plt.imshow(img)

# 腐蚀 卷积核框到黑色即变成黑色
erode = cv2.erode(img,(2,2),iterations=5)
plt.subplot(121),plt.imshow(erode)
# cv2.imshow('pic',erode)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# 膨胀  卷积核框到白色即变成白色
# 可以把被腐蚀影响到的图像还原回去
dilate = cv2.dilate(erode,(2,2),iterations=5)
plt.subplot(122),plt.imshow(dilate)
plt.show()
