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
    # 预处理
    gray_pic = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, threshold= cv2.threshold(gray_pic,20,255,cv2.THRESH_BINARY_INV)
    show_cv(threshold, "th_template")
    # 这里选择EXTERNAL 只检测外轮廓，SIMPLE只保留终点坐标
    tem_cnt, hierarchy = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, tem_cnt, -1, (0, 0, 255), 2)
    show_cv(img, "temple")
    # print(np.array(tem_cnt).shape)
    tem_cnt = sort_contours(tem_cnt, method="Left-to-right")[0]

    digits = {}
    for (i, c) in enumerate(tem_cnt):
        # 枚举结束后 （i,c）对应着（轮廓对应的数字，轮廓的索引）
        (x, y, w, h) = cv2.boundingRect(c)
        roi = threshold[y:y + h, x:x + w]  # 把矩形抠出来
        roi = cv2.resize(roi, (57, 88))  # resize成合适的大小

        digits[i] = roi
        # show_cv(roi, " ")

    return digits

def sort_contours(cnts, method="Left2Right"):
    reverse = False
    i = 0

    if method == "Left2Right" or method == "Bottom2Top":    reverse = True
    elif method=="Top2Bottom" or method == "Bottom2Top":   i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    #TODO: 未理解
    (cnts,boundingBoxes) = zip(*sorted(zip(cnts,boundingBoxes),
                                       key=lambda b:b[1][i],reverse=reverse))
    return cnts,boundingBoxes

def resize(img,length=0,width=0):
    """
    重设图像大小
    :param img: opencv格式图像
    :param length: 想要设置的长
    :param width: 想要设置的宽
    :return: 重设大小后的图像
    """
    if length == 0 and width==0:
        return img
    elif length==0 and width !=0:
        length = int((width / img.shape[0]) * img.shape[1])
        return cv2.resize(img,(length,width))
    elif length!=0 and width ==0:
        width = int((length / img.shape[1]) * img.shape[0])
        return cv2.resize(img, (length, width))
    else:   return cv2.resize(img, (length, width))

def img_pretreat(img,rectKernel,sqKernel):
    card = resize(img, width=300)
    gray_card = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    show_cv(gray_card, "card")
    # 礼冒操作，让数字更明显
    tophat = cv2.morphologyEx(gray_card, cv2.MORPH_TOPHAT, rectKernel)
    show_cv(tophat, "tophat")

    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=(3, 3))


if __name__ == '__main__':
    args = Set_argparse()
    temple = cv2.imread(args["template"])
    card = cv2.imread(args["image"])
    show_cv(temple,"template")

    # temple图像预处理
    digits = tem_pretreat(temple)

    # 初始化卷积核
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


    print()









