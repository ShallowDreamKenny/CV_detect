#!/usr/bin/env python3
# coding:utf-8
"""
# File       : 图像拼接.py
# Time       ：10/20/22 1:19 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import numpy as np
import matplotlib.pyplot as plt
from sitcher import Stitcher
import cv2

if __name__ == '__main__':
    img1 = cv2.imread("../../图像特征/2. 尺度空间/pic/TEST_1.jpg")
    img2 = cv2.imread("../../图像特征/2. 尺度空间/pic/TEST_2.jpg")
    img1 = cv2.resize(img1,(640,480))
    img2 = cv2.resize(img2, (640, 480))

    stitcher = Stitcher()
    result = stitcher.stitch([img1,img2])
    # cv2.imshow("re",result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()