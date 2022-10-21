"""
# File       : test_dynamic.py
# Time       ：2022/10/15 14:00
# Author     ：Kust Kenny
# version    ：python 3.8
# Description：
"""
import cv2

file = cv2.imread("./pic/east_str_hc.jpg")
file = cv2.cvtColor(file,cv2.COLOR_BGR2GRAY)
cap = cv2.VideoCapture("./video/car.mp4",)
# cv2.imshow("",file)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
_,basic = cap.read()
basic = cv2.resize(basic,(640,480))
gray = cv2.cvtColor(basic,cv2.COLOR_BGR2GRAY)
_,thre = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
basic = cv2.add(thre,file)
# basic = -basic
# cv2.imshow("",basic)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,thre = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
    # _,thre_f = cv2.threshold(file,80,255,cv2.THRESH_BINARY)
    thre = cv2.resize(thre,(640,480))
    add = cv2.add(thre,file)
    # add =
    end = cv2.absdiff(add,basic)
    basic = add
    if _:
        cv2.imshow(" ",end)
        cv2.imshow("f",frame)
        cv2.imshow("basic",basic)
        cv2.waitKey(30)
        print(cv2.countNonZero(end))