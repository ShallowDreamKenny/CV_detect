#!/usr/bin/env python3
# coding:utf-8
"""
# File       : tracking_d455.py
# Time       ：10/21/22 5:25 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import cv2
import numpy as np
import argparse


OPENCV_OBJECT_TRACKERS = {
    "kcf":cv2.TrackerKCF_create,
    "boost": cv2.legacy.TrackerBoosting_create,
    "csrt": cv2.TrackerCSRT_create,
    "goturn":cv2.TrackerGOTURN_create,
    "mil":cv2.TrackerMIL_create(),
    "mosse":cv2.legacy.TrackerMOSSE_create
}

def init_arg():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v","--video",type=str,default="./video/fly_1.mp4",help="input the video or camera")
    ap.add_argument("-t","--tracker",type=str,default="kcf",help="please choose the tracker algorithm")
    args = vars(ap.parse_args())
    return args

if __name__ == '__main__':
    args = init_arg()
    cap = cv2.VideoCapture(args["video"])
    trackers = cv2.legacy.MultiTracker_create()

    _ = True
    while _:
        _,frame = cap.read()

        (h,w) = frame.shape[:2]
        width = 600
        r = width/float(w)
        frame = cv2.resize(frame,(width,int(h*r)))

        (success,boxes) = trackers.update(frame)

        # 绘制区域
        try:
            for box in boxes:
                (x,y,w,h) = [int(v) for v in box]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        except:
            pass


        cv2.imshow("press to select a box",frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("s"):
            # 选择一个区域
            box = cv2.selectROI("press to select a box",frame,fromCenter=False,showCrosshair=True)

            # 创建一个新的追踪器
            tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
            trackers.add(cv2.legacy.TrackerKCF_create(),frame,box)
        elif key == 27:
            break
    cv2.destroyAllWindows()
    cap.release()