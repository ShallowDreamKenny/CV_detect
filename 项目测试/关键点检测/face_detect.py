#!/usr/bin/env python3
# coding:utf-8
"""
# File       : face_detect.py
# Time       ：10/24/22 4:45 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
    人脸关键点检测
"""
import cv2
import argparse
import dlib
import numpy as np
from collections import OrderedDict

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


def init_args():
    """
    初始化命令行参数
    :return:
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pic", default="./pic/liudehua.jpg")
    ap.add_argument("-m", "--model", default="./model/shape_predictor_68_face_landmarks.dat")
    return vars(ap.parse_args())

def init_pre(args):
    img = cv2.imread(args["pic"])
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["model"])
    return img,detector,predictor

def pre_process(img,width):
    """
    图像预处理
    :param img: 原始图像
    :return: 大小变换后的原始图像以及灰度图
    """
    (h, w) = img.shape[:2]
    # width = 500
    r = width/float(w)
    img = cv2.resize(img,(width,int(r*h)),interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img,gray


def shape_to_np(shape,dtype = "int"):
    coords = np.zeros((shape.num_parts,2),dtype=dtype)
    for i in range(0,shape.num_parts):
        coords[i] = (shape.part(i).x,shape.part(i).y)
    return coords


def visualize_output(img,shape,colors=None, alpha=0.75):
    overlay = img.copy()
    output = img.copy()

    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    for (i,name) in enumerate(FACIAL_LANDMARKS_68_IDXS.keys()):
        # 得到每一个点的坐标
        (j,k) = FACIAL_LANDMARKS_68_IDXS[name]
        pts = shape[j:k]
        # 检查位置
        if name == "jaw":
            for l in range(1,len(pts)):
                ptA = tuple(pts[l-1])
                ptB = tuple(pts[l])
                cv2.line(overlay,ptA,ptB,colors[i],2)
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay,[hull],-1,colors[i],-1)

    cv2.addWeighted(overlay,alpha,output,1-alpha,0,output)
    return output



if __name__ == '__main__':
    args = init_args()
    img, detector, predictor = init_pre(args)
    img, gray = pre_process(img,500)

    # 人脸检测
    rects = detector(gray,1)
    for (i,rect) in enumerate(rects):
        shape = predictor(gray,rect)
        coords = shape_to_np(shape)

        for (name,(i,j)) in FACIAL_LANDMARKS_68_IDXS.items():
            clone = img.copy()
            cv2.putText(clone,name,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

            for (x, y) in coords[i:j]:
                cv2.circle(clone,(x,y),3,(0,0,255),-1)

            (x,y,w,h) = cv2.boundingRect(np.array([coords[i:j]]))

            roi = img[y:y+h,x:x+w]
            roi = pre_process(roi,250)[0]

            cv2.imshow("roi",roi)
            cv2.imshow("Image",clone)
            cv2.waitKey(0)
        # 展示所有区域
        output = visualize_output(img,coords)
        cv2.imshow("output",output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()