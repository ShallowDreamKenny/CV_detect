#!/usr/bin/env python3
# coding:utf-8
"""
# File       : tracking_d455.py
# Time       ：10/21/22 5:25 PM
# Author     ：Kust Kenny
# version    ：python 3.6
# Description：
"""
import random

import cv2
import numpy as np
import argparse
import pyrealsense2 as rs
# import rospy
# from geometry_msgs.msg import Point





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
    ap.add_argument("-t","--tracker",type=str,default="kcf",help="please choose the tracker algorithm")
    args = vars(ap.parse_args())
    return args

if __name__ == '__main__':
    args = init_arg()
    trackers = cv2.legacy.MultiTracker_create()

    # rospy.init_node("listener", anonymous=True)
    # point = Point()
    # image_point_pubulish = rospy.Publisher('/camera/point', Point, queue_size=1)

    _ = True

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # 初始化nn
    weights_path = './models/Hpoint.weights'  # 模型权重文件
    cfg_path = './models/Hpoint.cfg'  # 模型配置文件
    labels_path = './models/Hpoint.names'  # 模型类别标签文件

    # 初始化一些参数
    LABELS = open(labels_path).read().strip().split("\n")
    Bs = []
    confidences = []
    classIDs = []
    color_list = []
    for i in range(len(LABELS)):
        color_list.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

    ln = net.getLayerNames()
    out = net.getUnconnectedOutLayers()  # 得到未连接层得序号  [[200] /n [267]  /n [400] ]
    x = []
    for i in out:  # 1=[200]
        x.append(ln[i - 1])  # i[0]-1    取out中的数字  [200][0]=200  ln(199)= 'yolo_82'
    ln = x

    while _:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        frame = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        (h,w) = frame.shape[:2]
        # width = 600
        # r = width/float(w)
        # frame = cv2.resize(frame,(width,int(h*r)))

        (success,boxes) = trackers.update(frame)

        # 绘制区域
        try:
            for box in boxes:
                (x,y,w,h) = [int(v) for v in box]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                a = 0
                i = 0
                for x1 in range(int(x+w-10),int(x+w+10)):
                    for y1 in range(int(y+w-10),int(y+w+10)):
                        a += depth_image[int(y1), int(x1)]
                        # a += depth_image[int(y1 + h / 2), int(x1 + w/2)]
                        #print(depth_image[int(y + h / 2), int(x + w/2)])
                        i += 1
                distance =  a/i
                # print(a / i)
                # point.x = x + w/2
                # point.y = y + h/2
                # point.z = distance
                # image_point_pubulish.publish(point)
        except:
            pass





        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            # trackers = cv2.legacy.MultiTracker_create()
            Bs = []
            confidences = []
            classIDs = []
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)
            for output in layerOutputs:  # 对三个输出层 循环
                for detection in output:  # 对每个输出层中的每个检测框循环
                    scores = detection[5:]  # detection=[x,y,h,w,c,class1,class2] scores取第6位至最后
                    classID = np.argmax(scores)  # np.argmax反馈最大值的索引
                    confidence = scores[classID]
                    if confidence > 0.5:  # 过滤掉那些置信度较小的检测结果
                        box = detection[0:4] * np.array([w, h, w, h])
                        # print(box)
                        (centerX, centerY, width, height) = box.astype("int")
                        # 边框的左上角
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        # 更新检测出来的框
                        Bs.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            try:
                idxs = cv2.dnn.NMSBoxes(Bs, confidences, 0.2, 0.3)
                box_seq = idxs.flatten()
                if len(idxs) > 0:
                    # pass
                    for seq in box_seq:
                        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
                        trackers.add(cv2.legacy.TrackerKCF_create(), frame, Bs[seq])
            except:
                print("detect None")
        elif key == 27:
            break
        cv2.imshow("press to select a box", frame)
        cv2.imshow("depth", depth_colormap)
    cv2.destroyAllWindows()
