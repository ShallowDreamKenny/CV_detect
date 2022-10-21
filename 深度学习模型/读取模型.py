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

net = cv2.dnn.readNetFromONNX("./models/test.onnx")
img = cv2.imread("./pic/img.png")

blob = cv2.dnn.blobFromImage(img,1,(32,32))

net.setInput(blob)
preds = net.forward()

idx = np.argsort(preds[0])[::-1][0]
print("类别为： ",idx)


