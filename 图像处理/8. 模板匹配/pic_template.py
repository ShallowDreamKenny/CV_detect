#!/usr/bin/env python3
# coding:utf-8
"""
# File       : pic_template.py
# Time       ：10/14/22 3:13 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread("pic/car.jpg")
logo = cv2.imread("pic/logo.jpg")
plt.subplot(121),plt.imshow(logo)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

"""
TM_SQDIFF: 计算平方不同，计算出来的值越小越相关
TM_CCORR: 计算相关性，计算出来的值越大，越相关
TM_CCOEFF: 计算相关系数，计算出来的值越大，越相关
TM_SQDIFF_NORMED： 计算归一化平方不同，计算出来的值越接近0，越相关
TM_CCORR_NORMED: 计算归一化相关性，计算出来的值越接近1，越相关
TM_CCOEFF_NORMED: 计算归一化相关系数，计算出来的值越接近1，越相关
"""
res = cv2.matchTemplate(img,logo,cv2.TM_SQDIFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
cv2.rectangle(img,(min_loc[0],min_loc[1]),(min_loc[0] + 114,min_loc[1] + 114),(0,255,0),2)
plt.subplot(122),plt.imshow(img)
plt.show()

