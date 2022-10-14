#!/usr/bin/env python3
# coding:utf-8
"""
# File       : Read_Video.py
# Time       ：10/12/22 6:48 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2

cap = cv2.VideoCapture("video_1.mp4")
if cap.isOpened():
    open, frame = cap.read()
else:
    open = False

while open:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('result',gray)
        if cv2.waitKey(30) & 0xFF==27:  #在waitkey(1)时延迟调低，会出现视频播放过快的情况
            break
    if frame is None:
        break
cap.release()
cv2.destroyAllWindows()

# video = cv2.imread("video_1.mp4")   对于视频流采用VideoCapture