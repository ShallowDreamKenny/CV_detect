#!/usr/bin/env python3
# coding:utf-8
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
import os
import cv2
import numpy as np
import time
IMAGE_WIDTH=640
IMAGE_HEIGHT=480

rospy.init_node("listener", anonymous=True)
image_pubulish=rospy.Publisher('/camera/image_raw',Image,queue_size=1)
def publish_image(imgdata):
    image_temp=Image()
    header = Header(stamp=rospy.Time.now())
    header.frame_id = 'map'
    image_temp.height=IMAGE_HEIGHT
    image_temp.width=IMAGE_WIDTH
    image_temp.encoding='rgb8'
    image_temp.data=np.array([imgdata]).tostring()
    print(image_temp.data)
    #image_temp.is_bigendian=True
    image_temp.header=header
    image_temp.step=640*3
    image_pubulish.publish(image_temp)

Video = cv2.VideoCapture('/home/kenny/Code/CV_Process/项目测试/目标追踪/video/fly_1.mp4')
# file_list=os.listdir('/home/kenny/Code/CV_Process/图像特征/1. 角点检测/pic')
# file_list.sort()
while True:
    # img=cv2.imread('/home/kenny/Code/CV_Process/图像特征/1. 角点检测/pic'+i)
    _, img = Video.read()
    if img is None:
        break
    img = cv2.resize(img,(640,480))
    publish_image(img)
    #time.sleep(1)
    # cv2.imshow('123',img)
    key=cv2.waitKey(10)
    if key==ord('q'):
        break

    # print("pubulish")
cv2.destroyAllWindows()