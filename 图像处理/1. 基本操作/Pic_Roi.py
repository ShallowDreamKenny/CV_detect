#!/usr/bin/env python3
# coding:utf-8
"""
# File       : Pic_Roi.py
# Time       ：10/12/22 6:57 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2

def cv_imshow(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread("cat.jpg")
# 提取3通道
b,g,r = cv2.split(img)
cat = img[200:800,300:600]
# cv_imshow("roi",cat)

print(b)
print(b.shape)
print(img.shape)

# 通道还原
bgr_pic = cv2.merge((b,g,r))
# cv_imshow("return",bgr_pic)

# 只保留一个通道
b_pic = bgr_pic.copy()
b_pic[:,:,0] = 0
b_pic[:,:,1] = 0
g_pic = bgr_pic.copy()
g_pic[:,:,0] = 0
g_pic[:,:,2] = 0
cv_imshow("b_pic",b_pic)
cv_imshow("g_pic",g_pic)