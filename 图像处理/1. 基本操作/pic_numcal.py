#!/usr/bin/env python3
# coding:utf-8
"""
# File       : pic_numcal.py
# Time       ：10/13/22 3:00 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2
from matplotlib import pyplot as plt

cat_image = cv2.imread("cat.jpg")
car_image = cv2.imread("car.jpg")

#3通道中每个像素点+10
car_image_2 = car_image + 10
# cv2.imshow("",car_image_2)

print(car_image.shape)
print(cat_image.shape)

res = cv2.resize(cat_image,(1280,720))
res_2 = cv2.resize(cat_image,(0,0),fx=1,fy=3)
res = cv2.addWeighted(res,0.1,car_image,0.1,100)
plt.imshow(res_2)

cv2.waitKey(0)

plt.show()

