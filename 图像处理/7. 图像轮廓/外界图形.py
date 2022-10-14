#!/usr/bin/env python3
# coding:utf-8
"""
# File       : 外界图形.py
# Time       ：10/14/22 2:47 PM
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

#外接矩形
img2 = img.copy()
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
plt.subplot(131),plt.imshow(img2)

#外接圆
img3 = img.copy()
(x,y),radius = cv2.minEnclosingCircle(cnt)
cv2.circle(img3,(int(x),int(y)),int(radius),(0,255,0),2)
plt.subplot(132),plt.imshow(img3)

plt.show()
