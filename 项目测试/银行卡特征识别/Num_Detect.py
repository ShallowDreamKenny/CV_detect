#!/usr/bin/env python3
# coding:utf-8
"""
# File       : Num_Detect.py
# Time       ：10/14/22 5:26 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：银行卡数字识别，并可将同样的方法移植到车牌识别当中去
#TODO： 这里之后补充整个项目实现的流程
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def Set_argparse():
    """
    初始化参数
    :return: 参数的值
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default="./pic/card1.png",help="path of the input image")
    ap.add_argument("-t", "--template", default="./pic/model.png",help="path of the input model")
    args = vars(ap.parse_args())
    return args

def show_cv(img,name):
    """
    使用opencv的方法显示图像，按任意键退出图像并继续执行程序
    :param img: opencv类型图像（np.array）
    :param name: 显示图像的窗口名称
    :return: None
    """
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def tem_pretreat(img):
    """
    对于模板图像进行预处理
    :param img: 模板图像 （BGR格式）
    :return: 模板的灰度图像
    """
    gray_pic = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, threshold= cv2.threshold(img,20,255,cv2.THRESH_BINARY_INV)
    return threshold

if __name__ == '__main__':
    args = Set_argparse()
    temple = cv2.imread(args["template"])
    show_cv(temple,"template")

    th_tem = tem_pretreat(temple)
    show_cv(th_tem,"th_template")

