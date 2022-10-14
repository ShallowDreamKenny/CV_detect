#!/usr/bin/env python3
# coding:utf-8
"""
# File       : 多结果匹配.py
# Time       ：10/14/22 3:30 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("pic/mario.jpg")
logo = cv2.imread("pic/coin.jpg")
plt.subplot(121),plt.imshow(logo)

res = cv2.matchTemplate(img,logo,cv2.TM_CCOEFF_NORMED)

threshold = 0.8
loc = np.where(res >= threshold)
for point in zip(*loc[::-1]):
    # print(point)
    bottom = (point[0] + 133, point[1] + 133)
    cv2.rectangle(img,point,bottom,(0,255,0),1)
plt.subplot(122),plt.imshow(img)
plt.show()
