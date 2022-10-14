#!/usr/bin/env python3
# coding:utf-8
"""
# File       : Pic_boundary_fill.py
# Time       ：10/12/22 7:07 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
# plt.rcParams["font.sans-serif"]=["SimHei"]
# plt.rcParams["axes.unicode_minus"]=False
top_size, bottom_size, left_size, reght_size = (50,50,50,50)
img = cv2.imread("car.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

replicate = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,reght_size,borderType=cv2.BORDER_REPLICATE)
repflect = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,reght_size,borderType=cv2.BORDER_REFLECT)
replicate_101 = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,reght_size,borderType=cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,reght_size,borderType=cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,reght_size,borderType=cv2.BORDER_CONSTANT,value=255)

plt.subplot(231),plt.imshow(img,'gray'),plt.title('ORIGINAL')
plt.subplot(232) , plt.imshow(replicate,'gray'), plt.title('REPLICATE')
plt.subplot(233) , plt.imshow(replicate_101,'gray'), plt.title('REPLICATE_101')
plt.subplot(234) , plt.imshow(repflect,'gray'), plt.title('REFLECT')
plt.subplot(235) , plt.imshow(wrap,'gray'), plt.title('WRAP')
plt.subplot(236) , plt.imshow(constant,'gray'), plt.title('CONSTANT')

# #五种边缘填充方法
# REFLECT:反射法
# REPLICATE:复制法
# CONSTANT:常数法
plt.show()
