#!/usr/bin/env python3
# coding:utf-8
"""
# File       : 读取模型.py
# Time       ：10/21/22 4:38 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2.dnn
import numpy as np

# weights模型需要导入cfg文件，并且需要opencv的版本在4.4以上
# 推荐3.7版本的python安装4.5.1版本opencv-python及opencv-contrib-python
net = cv2.dnn.readNetFromDarknet("./models/yolov4s.cfg","./models/yolov4s.weights")
img = cv2.imread("./pic/img.png")

blob = cv2.dnn.blobFromImage(img,1,(32,32))

net.setInput(blob)
preds = net.forward()

idx = np.argsort(preds[0])[::-1][0]
print("类别为： ",idx)


