#!/usr/bin/env python3
# coding:utf-8
"""
# File       : Read_pic.py
# Time       ：10/12/22 6:34 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np


def cv_imshow(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('car.jpg')  #opencv默认为读取BGR
cv_imshow("car",img)

img_gray = cv2.imread("car.jpg", cv2.IMREAD_GRAYSCALE)
cv_imshow("GRAY_CAR",img_gray)

#保存图像
cv2.imwrite("gray_car",img_gray)
# print(img.shape)