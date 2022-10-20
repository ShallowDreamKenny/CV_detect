#!/usr/bin/env python3
# coding:utf-8
"""
# File       : sitcher.py
# Time       ：10/20/22 1:35 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import numpy as np
import cv2

class Stitcher:
    # 拼接函数
    def stitch(self,imgs,ratio = 0.75, reprojThresh = 4.0,showMatches = True):
        (img1,img2) = imgs

        # 检测两张图片的关键点
        (kps1,features1) = self.detectAndDescribe(img1)
        (kps2,features2) = self.detectAndDescribe(img2)

        # 匹配两张图片的所有特征点，返回匹配结果
        M = self.matchKeypoints(kps1,kps2,features1,features2,ratio,reprojThresh=reprojThresh)
        if M is None:
            return None

        # 提取匹配结果
        (matches,H,status) = M
        # 将图像A进行视角变换
        result = cv2.warpPerspective(img1,H,(img1.shape[1]+img2.shape[1],img1.shape[0]))
        # result = cv2.drawMatchesKnn(img1, kps1, img2, kps2, status, None, flags=2)
        self.cv_show("result",result)
        result[0:img2.shape[0],0:img2.shape[1]] = img2
        self.cv_show("result", result)
        return result

    # 得到关键特征点
    def detectAndDescribe(self,img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # 建立orb
        orb = cv2.ORB_create()
        (kps,features) = orb.detectAndCompute(img,None)

        # 把结果转换成Numpy数组
        kps = np.float32([kp.pt for kp in kps])

        return (kps,features)

    # 匹配特征点
    def matchKeypoints(self,kps1,kps2,features1,features2,ratio,reprojThresh):
        match = cv2.BFMatcher()
        rawMatchs = match.knnMatch(features1,features2,2)

        matches = []
        for m in rawMatchs:
            # 判断： 当最近距离跟次近距离的比值小于ratio值时，保留次配对比
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx,m[0].queryIdx))

        # 当筛选后的匹配数据大于4时，计算视角变换矩阵
        if len(matches)>4:
            # 获取匹配对的点坐标
            pst1 = np.float32([kps1[i] for (_,i) in matches])
            pst2 = np.float32([kps2[i] for (i,_) in matches])

            # 计算视角变换矩阵
            (H,status) = cv2.findHomography(pst1,pst2,cv2.RANSAC,reprojThresh)

            return (matches,H,status)

    def cv_show(self, name, img):
        cv2.imshow(name,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()