#!/usr/bin/env python3
# coding:utf-8
"""
# File       : pic_开闭运算.py
# Time       ：10/13/22 4:21 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2
import numpy as np

img = cv2.imread("pic_erode.jpg")
#开运算，先腐蚀再膨胀
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,(3,3),iterations=5)
#闭运算，先膨胀再腐蚀
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,(3,3),iterations=5)

res = np.hstack((img,opening,closing))
cv2.imshow("",res)
cv2.waitKey(0)
cv2.destroyAllWindows()