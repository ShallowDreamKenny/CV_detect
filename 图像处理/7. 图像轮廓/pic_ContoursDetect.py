#!/usr/bin/env python3
# coding:utf-8
"""
# File       : pic_ContoursDetect.py
# Time       ：10/13/22 9:33 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
cv2.findContours(img,mode,method)

mode: 轮廓检索模式
    1. RETR_EXTERNAL :  只检测最外面的轮廓
    2. RETR_LIST: 检索所有的轮廓，并将其保存到一条链表当中；
    3  RETR_CCOMP: 检索所有轮廓，并将它们组织为两层；，顶层十个部分的外部边界，第二层是空洞的边界
    4. RETR_TREE:  检索所有的轮廓，并重构嵌套轮廓的整个层次  最常用！！！！！！

method： 轮廓逼近方法（常用的两种）
    1. CHAIN_APPROX_NONE: 以Freeman编码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）。
    2. CHAIN_APPROX_SIMPLE: 压缩水平的、、垂直的和斜的部分，也就是函数值保留他们的重点部分
"""
def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread("pic/car.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# cv_show(thresh,"thresh")
plt.imshow(thresh)
# binary, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
"""
binary: 二值化后的图像  （新版本不再返回binary）
contours： 返回的所有轮廓点
hierarchy： 
"""
draw_img = img.copy()
#-1为所有轮廓,其他值为制定轮廓
res = cv2.drawContours(draw_img,contours,-1,(0,0,255),2)
res_2 = cv2.drawContours(img,contours,100,(255,0,255),2)
res = np.hstack((res,res_2))
# cv_show(res,"res")
plt.imshow(res)

# 轮廓特征
cnt = contours[150]
aera = cv2.contourArea(cnt)
length = cv2.arcLength(cnt,True)  # True表示闭合

plt.show()
