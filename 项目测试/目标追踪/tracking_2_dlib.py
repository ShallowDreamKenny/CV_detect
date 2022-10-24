#!/usr/bin/env python3
# coding:utf-8
"""
# File       : tracking_2_dlib.py
# Time       ：10/24/22 1:06 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：先导入深度学习框架，检测出boundingbox后再，使用dlib进行追踪
"""
from FPS_Class import FPS
import cv2
import numpy as np
import dlib
import argparse


def init_arg():
    """
    初始化命令行参数
    :return: 参数
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", default="./model/MobileNetSSD_deploy.prototxt", help="path to the prototxt")
    ap.add_argument("-m", "--model", default="./model/MobileNetSSD_deploy.caffemodel", help="path to the model")
    ap.add_argument("-v", "--video", default="./video/person.mp4", help="path to the video")
    ap.add_argument("-o", "--output", type=str, help="path to optinal output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detecttion")

    return vars(ap.parse_args())


# 标签
CLASS = ["background", "aeroplane", "bicycle", "bird", "boat",
         "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
         "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
         "sofa", "train", "tvmonitor"]

def init_pre(args):
    """
    初始化模型和视频流
    :param args: 命令航参数
    :return:
        1. 视频流
        2. dnn网络
    """
    print("[INFO] Loading video...")
    cap = cv2.VideoCapture(args["video"])
    print("[INFO] Loading model...")
    net = cv2.dnn.readNet(args["prototxt"],args["model"])
    return cap,net

# 预处理操作
def pre_process_frame(img):
    """
    初始化图像
    :param img: 原图
    :return: rgb格式图
    """
    (h,w) = img.shape[:2]
    width = 600
    r = width / float(w)
    dim = (width,int(h*r))
    img = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img,rgb

if __name__ == '__main__':
    args = init_arg()
    cap, net = init_pre(args)
    writer = None

    # 追踪目标的数据
    trackers = []
    labels = []

    fps = FPS().start()

    while True:
        # 读取一帧
        _,frame = cap.read()
        if frame is None:
            break

        frame,rgb = pre_process_frame(frame)

        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"],fourcc,30,
                                     (frame.shape[1],frame.shape[0]),True)
        # elif args["output"] is None and writer is None:
        #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        #     writer = cv2.VideoWriter("./output/output.MJPG", fourcc, 30,
        #                              (frame.shape[1], frame.shape[0]), True)

        if len(trackers) == 0:
            (h,w) = frame.shape[:2]
            # 与模型网络保持一致
            blob = cv2.dnn.blobFromImage(frame,0.007843,(w,h),127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0,detections.shape[2]):
                # 根据不同框架调整以下内容
                # 第i个检测结果的概率值
                confidence = detections[0,0,i,2]
                if confidence > args["confidence"]:
                    # 第i个检测结果的id值
                    idx = int(detections[0,0,i,1])
                    label = CLASS[idx]

                    # 只保留人的
                    if label != "person":
                        continue

                    # 得到boundingbox
                    # 根据框架修改的得到的box值
                    # 该框架得到的是相对位置，因此需要分别×w和h得到真实坐标值
                    box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                    # dlib不支持小数
                    (startX,startY,endX,endY) = box.astype("int")

                    # 使用dlib进行目标追踪
                    t = dlib.correlation_tracker()
                    rect = dlib.rectangle(int(startX),int(startY),int(endX),int(endY))
                    t.start_track(rgb,rect)

                    # 保存结果
                    labels.append(label)
                    trackers.append(t)
                    cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),2)
                    cv2.putText(frame,label,(startX,startY-15),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)
        else:#已经有框之后
            for(t,l) in zip(trackers,labels):
                t.update(rgb)
                pos = t.get_position()

                # 得到位置
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),2)
                cv2.putText(frame, l, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("frame",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        fps.update()
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    if writer is not None:
        writer.release()

    cap.release()
    cv2.destroyAllWindows()

