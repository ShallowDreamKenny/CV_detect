#!/usr/bin/env python3
# coding:utf-8
"""
# File       : tracking_high_opencv.py
# Time       ：10/24/22 6:00 PM
# Author     ：Kust Kenny
# version    ：python 3.6 opencv 4.6.0.66
# Description：
"""
import cv2
# a = cv2.TrackerKCF_create()
a = cv2.legacy.MultiTracker_create()
# cap = cv2.VideoCapture("video/fly_1.mp4")
cap = cv2.VideoCapture("http://192.168.137.169:8081")


_,frame = cap.read()
# roi = cv2.selectROI("tracker",frame)
# a.init(frame,roi)
b = 0
# c = cv2.legacy.TrackerKCF()
while _:
    _,frame = cap.read()
    if b ==1:
        t = a.update(frame)[1]
        for i,(x,y,w,h) in enumerate(t):
        # (x,y,w,h) = t
            frame = cv2.rectangle(frame,(int(x),int(y)),(int(x+w),int(y+h)),(0,0,255),2)
            print(i,":",(x,y,w,h))

    k = cv2.waitKey(10)
    if k == ord("a"):
        roi = cv2.selectROI("tracker", frame)
        print(roi)

        # a.init(frame, roi)
        a.add(cv2.legacy.TrackerKCF_create(),frame, roi)
            # t = a.update(frame)
        # except:
        #     pass
        b = 1
    elif k == 27:
        break
    cv2.imshow("tracker",frame)

cap.release()
cv2.destroyAllWindows()