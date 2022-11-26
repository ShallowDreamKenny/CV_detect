#!/usr/bin/env python3
# coding:utf-8
"""
# File       : test_dynamic_car.py
# Time       ：10/27/22 4:06 PM
# Author     ：Kust Kenny
# version    ：python 3.6 opencv 4.6.0.66
# Description：
"""
import cv2

file = cv2.imread("./pic/east_str_hc.jpg")
# file = cv2.cvtColor(file,cv2.COLOR_BGR2GRAY)
cap = cv2.VideoCapture("./video/car.mp4",)


fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    _,frame = cap.read()
    if _:
        frame = cv2.resize(frame,(640,480))
        add = cv2.add(frame,file)
        add = fgbg.apply(add)

        cv2.imshow(" ",add)
        cv2.imshow("f",frame)
        cv2.waitKey(30)
        print(cv2.countNonZero(add))
    else:
        break
cv2.destroyAllWindows()
cap.release()