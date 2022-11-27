#!/usr/bin/env python3
# coding:utf-8
"""
# File       : 接受图像.py
# Time       ：11/26/22 8:59 PM
# Author     ：Kust Kenny
# version    ：python 3.6 
# Description：
"""
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

cvbri = CvBridge()
def callback(data):
    # define picture to_down' coefficient of ratio
    cv_image = cvbri.imgmsg_to_cv2(img_msg=data, desired_encoding="passthrough")
    scaling_factor = 0.5
    global count, bridge
    count = count + 1
    if count == 1:
        count = 0
        # cv_img = np.array(data)
        cv2.imshow("frame", cv_image)
        cv2.waitKey(3)
    else:
        pass


def displayWebcam():
    rospy.init_node('webcam_display', anonymous=True)

    # make a video_object and init the video object
    global count, bridge
    count = 0
    bridge = CvBridge()
    rospy.Subscriber('/camera/image_raw', Image, callback)
    rospy.spin()


if __name__ == '__main__':
    displayWebcam()