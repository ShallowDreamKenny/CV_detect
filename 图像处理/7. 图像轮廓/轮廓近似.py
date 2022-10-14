#!/usr/bin/env python3
# coding:utf-8
"""
# File       : 轮廓近似.py
# Time       ：10/14/22 2:31 PM
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

img = cv2.imread("pic/pic_2.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

draw_img = img.copy()
cnt = contours[19]
res = cv2.drawContours(draw_img,[cnt],-1,(255,0,255),2)
plt.imshow(res)

epsilon = 0.03 * cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
img_2 = img.copy()
res_2 = cv2.drawContours(img_2,[approx],-1,(255,0,255),2)
plt.imshow(res_2)
plt.show()
