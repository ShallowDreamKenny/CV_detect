#!/usr/bin/env python3
# coding:utf-8
"""
# File       : Detect.py
# Time       ：10/20/22 3:45 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：对于答题卡中涂的卡进行识别，并返回答题的正确率
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# 正确的答案
answer = {0:1,
          1:4,
          2:0,
          3:3,
          4:1}

def sort_contours(cnts, method="Left2Right"):
    """
    对于contours进行排序
    :param cnts: 输入的contours
    :param method: 排序的方法
    :return: 排完序之后的contours
    """
    reverse = False
    i = 0

    if method == "Left2Right" or method == "Bottom2Top":    reverse = True
    elif method=="Top2Bottom" or method == "Bottom2Top":   i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts,boundingBoxes) = zip(*sorted(zip(cnts,boundingBoxes),
                                       key=lambda b:b[1][i],reverse=reverse))
    return cnts,boundingBoxes

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

def Init_arg():
    """
    初始化外部参数
    :return:外部参数类型
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default="./pic/test_01.png",
                    help="path of the input image")
    ap.add_argument("-t", "--total", default=5,
                    help="total topic number")
    args = vars(ap.parse_args())
    return args

def img_preprocess(img):
    """
    答题卡图像预处理
    :param img: 答题卡原图
    :return: 1. 边缘图像 2. 灰度图像
    """
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    # show_cv(blurred,"blurred")
    edge = cv2.Canny(blurred,75,200)
    # show_cv(edge,"edge")
    plt.subplot(121),plt.imshow(blurred,"gray"),plt.title("blurred")
    plt.subplot(122),plt.imshow(edge,"gray"),plt.title("edge")
    plt.show()

    return edge,gray

def get_docCnt(img,edge):
    """
    获取答题卡目标区域
    :param img: 原图像
    :param edge: 原图像的边缘检测图像
    :return: 答题卡目标区域框
    """
    contours_img = img.copy()
    # 轮廓检测，透视变换
    cnts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[1]
    cv2.drawContours(contours_img, cnts, -1, (0, 0, 255), 3)
    # show_cv(contours_img, "contours")
    plt.imshow(contours_img)
    plt.show()
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            # 做一个多边形近似
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                docCnt = approx
                break
    return docCnt

def order_points(pts):
    # 一共四个坐标点
    rect = np.zeros((4,2),dtype="float32")
    # 按顺序找到对应坐标0123分别是左上，右上，右下，左下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def Perspective_transformation(gray_img,docCnt):
    """
    变换答题卡图像，随手拍的一张图向里可能并不只包含答题卡，还有其他物品，因此我们需要通过图像变换得到只有答题卡的一张图乡
    :param gray_img:  原始图片灰度图
    :param docCnt:  满足长方形的多边形拟合框
    :return:  变换后的答题卡图像
    """
    rect = order_points(docCnt)
    (tl,tr,br,bl) = rect

    # 计算输入的w和h值
    widthA = np.sqrt(((br[0]-bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA),int(widthB))

    heightA = np.sqrt(((tr[0]-br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA),int(heightB))

    # 变换后对应的坐标位置
    dst = np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth-1,maxHeight-1],
        [0,maxHeight-1]
    ],dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect,dst)
    warped = cv2.warpPerspective(gray_img,M,(maxWidth,maxHeight))
    # show_cv(warped,"wa")
    plt.imshow(warped,"gray"),plt.title("warped_img")
    plt.show()

    return warped

def filtrate_cnts(warped):
    """
    因为有着特殊字迹及障碍，我们需要筛选候框，只余下答题区域的框
    :param warped: 变换后的答题卡图像
    :return: 1. 筛选后得到的答题卡候选框 2. 变换后的答题卡二值化图像
    """
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    plt.subplot(121), plt.imshow(thresh, "gray"), plt.title("thresh_img")

    thresh_contours = thresh.copy()
    # 找到每一个圆圈轮廓
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    thresh_contours = cv2.cvtColor(thresh_contours, cv2.COLOR_GRAY2RGB)

    # 筛选轮廓，过滤掉
    questionCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # 根据实际情况制定标准
        if (w >= 20 and h >= 20 and ar >= 0.9 and ar < 1.1):
            questionCnts.append(c)
    questionCnts = sort_contours(questionCnts, "Top2Bottom")[0]
    cv2.drawContours(thresh_contours, questionCnts, -1, (255, 0, 0), 3)
    # show_cv(thresh_contours, "thresh_contours")
    plt.subplot(122), plt.imshow(thresh_contours), plt.title("thresh_contours")
    plt.show()
    return questionCnts,thresh

def circle_answer(warped,questionCnts,thresh):
    """
    将图像进行对比
    :param warped:  变换后的答题卡图像
    :param questionCnts: 过滤后的目标位置候选框
    :param thresh:  变换后的答题卡二值化图想（用作比对）
    :return: 1.比对完成标记过的图像 2. 正确的个数
    """
    # 重新二值化透视变换后的图像
    warped_color = warped.copy()
    warped_color = cv2.cvtColor(warped_color, cv2.COLOR_GRAY2BGR)

    # 进行判断
    correct = 0
    # 选项判断识别
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
        # 排序
        cnts = sort_contours(questionCnts[i:i + 5])[0]
        bubbled = None
        cnts = [cnts[4 - i] for (i, k) in enumerate(cnts)]
        # 遍历每一个结果
        for (j, c) in enumerate(cnts):
            # 使用mask来判断结果
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)  # 最后一个-1表示填充

            # 通过计算非零点数量来算是否选择这个答案
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            # 发现600以上一般为能被识别到
            total = cv2.countNonZero(mask)
            # show_cv(mask,"mask")
            plt.subplot(150 + j + 1), plt.imshow(mask, "gray")

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)
        plt.show()
        # 对比正确答案
        color = (0, 0, 255)
        k = answer[q]
        if k == bubbled[1]:
            correct += 1
            color = (0, 255, 0)
        cv2.drawContours(warped_color, [cnts[k]], -1, color, 3)

    return warped_color,correct


if __name__ == '__main__':
    args = Init_arg()
    img = cv2.imread(args["image"])
    # 预处理原图像
    edge,gray = img_preprocess(img)

    # 得到答题卡位置
    docCnt = get_docCnt(img,edge)
    # 重新变换图像（透视变换）
    warped =  Perspective_transformation(gray,docCnt.reshape(4,2))

    # 拿到筛选后的contours
    questionCnts,thresh = filtrate_cnts(warped)
    # 判断答案是否正确
    warped_color,corrects = circle_answer(warped,questionCnts,thresh)
    show_cv(warped_color," ")
    print("正确率为：{}%".format(corrects/args["total"]*100))


