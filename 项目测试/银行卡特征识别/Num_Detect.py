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
import matplotlib
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
    show_cv(img,"temple")
    gray_pic = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, threshold= cv2.threshold(gray_pic,20,255,cv2.THRESH_BINARY_INV)
    plt.subplot(131), plt.imshow(gray_pic, "gray"), plt.title("gray_temple")
    plt.subplot(132), plt.imshow(threshold, "gray"), plt.title("threshold_temple")
    # show_cv(threshold, "th_template")


    # 这里选择EXTERNAL 只检测外轮廓，SIMPLE只保留终点坐标
    tem_cnt, hierarchy = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, tem_cnt, -1, (0, 0, 255), 2)
    # show_cv(img, "temple")
    plt.subplot(133), plt.imshow(img), plt.title("temple")
    plt.show()

    # print(np.array(tem_cnt).shape)
    tem_cnt = sort_contours(tem_cnt, method="Left-to-right")[0]

    digits = {}
    for (i, c) in enumerate(tem_cnt):
        # 枚举结束后 （i,c）对应着（轮廓对应的数字，轮廓的索引）
        (x, y, w, h) = cv2.boundingRect(c)
        roi = threshold[y:y + h, x:x + w]  # 把矩形抠出来
        roi = cv2.resize(roi, (57, 88))  # resize成合适的大小

        digits[i] = roi
        # TODO:这里不知道如何显示超过是个数的图像
        # plt.subplot(25),plt.subplot(i+1), plt.imshow(roi,"gray"), plt.title(str(i))
    # plt.show()
    return digits

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

def temple_mapping(card,threshold_card,gray_card,digits,):
    """
    对模板和提取出来的数字进行匹配，并得到最终的银行卡号
    :param card: 银行卡原图
    :param threshold_card:  银行卡二值化图
    :param gray_card:   银行卡灰度图
    :param digits:  模板数字
    :return:得到的数字数组
    """
    cnts, hierarchy = cv2.findContours(threshold_card.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt_1 = cnts
    cur_card = card.copy()
    cv2.drawContours(cur_card, cnt_1, -1, (0, 0, 255), 3)
    plt.subplot(121), plt.imshow(cur_card, "gray"), plt.title("all_contours")
    # show_cv(cur_card, "countors")

    # 操作后发现轮廓太杂，需要进行筛选
    locs = []
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        # 计算长宽比
        ar = float(w) / h
        # x : 44-120 y: 158 - 183   比例约为2.666
        # 实测约为3
        if 3 < ar < 3.5:
            if 70 < w < 90:
                locs.append((x, y, w, h))
            # cv2.drawContours(card, c,-1, (0, 0, 255), 3)
    locs = sorted(locs, key=lambda x: x[0])
    # 画出最终所得的所有轮廓
    cur_card = card.copy()
    for loc in locs:
        # print(loc[2])
        cv2.rectangle(cur_card, (loc[0], loc[1]), (loc[0] + loc[2], loc[1] + loc[3]), (0, 255, 0), 2)
    # show_cv(cur_card, "countors")
    plt.subplot(122), plt.imshow(cur_card, "gray"), plt.title("result_contours")
    plt.show()

    output = []
    # 对每一组数字进行匹配
    for (i, (gx, gy, gw, gh)) in enumerate(locs):
        groupOutPut = []
        # 把每组数据提取出来
        group = gray_card[gy - 5:gy + gh + 5, gx - 5:gx + gw + 5]
        plt.subplot(240+int(i+1)), plt.imshow(group, "gray"),\
        plt.title("number_origin_"+str(i+1),size = 6)
        # show_cv(group, "group")

        # 对提取出来的数据进行处理
        _, group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        plt.subplot(240 + int(i+5)), plt.imshow(group, "gray")
        plt.title("number_threshold_"+str(i+1),size = 6)
        # show_cv(group, "thresh_group")



        # 计算轮廓
        group_cnts, hierarchy = cv2.findContours(group, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ditCnts = sort_contours(group_cnts, method="Left2Right")[0]

        # 测试绘画
        # cur_group = group.copy()
        # cv2.drawContours(cur_group,ditCnts,-1,(0,0,255),1)
        # show_cv(cur_group,"num")

        # 计算每一组中的每一个数值
        for cnt in ditCnts:
            (x, y, w, h) = cv2.boundingRect(cnt)
            roi = group[y:y + h, x:x + w]
            roi = resize(roi, 57, 88)
            # show_cv(roi, "roi")

            # 计算匹配得分
            scores = []

            for (digit, digitROI) in digits.items():
                result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF_NORMED)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)
            groupOutPut.append(str(np.argmax(scores)))
        # 发现列表顺序反了，原因未知
        groupOutPut.reverse()
        print(groupOutPut)

        #         在银行卡中画出来
        cv2.rectangle(card, (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5), (0, 0, 255), 1)
        cv2.putText(card, "".join(groupOutPut), (gx, gy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

        output.extend(groupOutPut)
    plt.show()
    return output

def img_pretreat(img,rectKernel,sqKernel):
    """
    对银行卡图像进行预处理
    :param img: 传入银行卡图像 BGR格式图
    :param rectKernel: 卷积核1
    :param sqKernel: 卷积核2
    :return:
            card: 重置大小后的原图像
            threshold:银行卡处理后的图像，只保留有数值区域
            gray_card: 银行卡灰度图像，以便于后续模板匹配
    """
    card = resize(img, width=300)
    gray_card = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    plt.subplot(231),plt.imshow(gray_card,"gray"),plt.title("gray_card")
    # show_cv(gray_card, "card")

    # 礼冒操作，让数字更明显
    tophat = cv2.morphologyEx(gray_card, cv2.MORPH_TOPHAT, rectKernel)
    # show_cv(tophat, "tophat")
    plt.subplot(232), plt.imshow(tophat, "gray"), plt.title("tophat")

    # 用sobel算子进行边缘检测
    # -1 默认为3*3卷积核
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = cv2.convertScaleAbs(gradX)

    gradY = cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=0,dy=1,ksize=-1)
    gradY = cv2.convertScaleAbs(gradY)
    # show_cv(gradX,"gradX")
    # show_cv(gradY,"gradY")

    # 这里只用X算子，因为用XY两方向算子得出来图像存在误差
    gradXY = gradX
    # gradXY = cv2.add(gradY,gradX)
    # show_cv(gradXY, "gradXY")
    plt.subplot(233), plt.imshow(gradXY, "gray"), plt.title("gradXY")

    # 出来的图像需要解析出我们所需要的区域，因此我们进行一个闭操作
    # 经过多次尝试，发现迭代3次效果较好
    gradXY = cv2.morphologyEx(gradXY,cv2.MORPH_CLOSE,rectKernel,iterations=3)
    # show_cv(gradXY, "gradXY")
    plt.subplot(234), plt.imshow(gradXY, "gray"), plt.title("Close_1")

    _,threshold = cv2.threshold(gradXY,230,255,cv2.THRESH_BINARY)
    # 双峰图像可以用OSTU来自适应阈值，0表示自适应，本实验采用双峰阈值效果不佳
    # _,threshold = cv2.threshold(gradXY,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    # show_cv(threshold,"threshold")
    plt.subplot(235), plt.imshow(threshold, "gray"), plt.title("Threshold")

    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, sqKernel, iterations=3)
    plt.subplot(236), plt.imshow(threshold, "gray"), plt.title("Close_2")
    show_cv(threshold, "threshold")

    plt.show()


    return card,gray_card,threshold

if __name__ == '__main__':
    args = Set_argparse()
    temple = cv2.imread(args["template"])
    card = cv2.imread(args["image"])

    # temple图像预处理
    digits = tem_pretreat(temple)

    # 初始化卷积核
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 对银行卡图像进行预处理
    card,gray_card,threshold_card = img_pretreat(card,rectKernel,sqKernel)

    # 模版匹配
    output = temple_mapping(card,threshold_card,gray_card,digits)

    #打印号码
    show_cv(card,"num")
    number = ""
    for num in output:
        number = number + str(num)
    print("该银行卡的卡号为：{}".format(number))









