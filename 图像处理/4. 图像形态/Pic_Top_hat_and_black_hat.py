#!/usr/bin/env python3
# coding:utf-8
"""
# File       : Pic_Top_hat_and_black_hat.py
# Time       ：10/13/22 4:47 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2
import numpy as np

# 礼帽 TOPHAT
# 原始输入 - 开运算结果(先腐蚀再膨胀)
# 得到毛刺

# 黑帽 BLACKHAT
# 闭运算（先膨胀再腐蚀） - 原始输入
# 得到轮廓

img = cv2.imread("pic_erode.jpg")
tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,(2,2),iterations=2)
blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,(2,2),iterations=2)

res = np.hstack((img,tophat,blackhat))
cv2.imshow("res",res)
cv2.waitKey(0)
cv2.destroyAllWindows()
