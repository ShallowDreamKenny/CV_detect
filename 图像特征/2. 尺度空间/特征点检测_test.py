#!/usr/bin/env python3
# coding:utf-8
"""
# File       : 特征点检测_test.py
# Time       ：10/24/22 6:10 PM
# Author     ：Kust Kenny
# version    ：python 3.6 opencv 4.6.0.66
# Description：
"""
import cv2
import numpy as np

if __name__ == '__main__':
    img1 = cv2.imread("./pic/img.png")
    img2 = cv2.imread("./pic/img2.png")


    ORB = cv2.ORB_create()
    sift = cv2.SIFT_create()
    kp1, des1 = ORB.detectAndCompute(img1,None)
    kp2, des2 = ORB.detectAndCompute(img2,None)

    kp3, des3 = sift.detectAndCompute(img1,None)
    kp4, des4 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches1 = bf.knnMatch(des1,des2,k=2)
    matches2 = bf.knnMatch(des3,des4,k=2)
    good1 = []
    good2 = []
    for ((m,n),(a,b)) in zip(matches1,matches2):
        if m.distance<0.75*n.distance:
            good1.append([m])
        if a.distance<0.75*b.distance:
            good2.append([a])

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good1,None,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    img4 = cv2.drawMatchesKnn(img1, kp3, img2, kp4, good2, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    img5 = np.hstack([img3,img4])
    cv2.imshow("img", img5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


