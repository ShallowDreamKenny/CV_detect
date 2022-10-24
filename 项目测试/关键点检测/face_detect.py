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

def pre_process(img):
    """
    图像预处理
    :param img: 原始图像
    :return: 大小变换后的原始图像以及灰度图
    """
    (h, w) = img.shape[:2]
    width = 500
    r = width/w
    img = cv2.resize(img,(width,int(r*h)))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img,gray

if __name__ == '__main__':
    args = init_args()
    img, detector, predictor = init_pre(args)
    img, gray = pre_process(img)

