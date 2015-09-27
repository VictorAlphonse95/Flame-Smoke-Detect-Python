#! /usr/bin/env python
#coding=utf-8
import cv2
import numpy as np

vName = '../video/forest1.avi'

cap = cv2.VideoCapture(vName)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
    if frame == None:
        print '视频读取完毕'
        break


    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    cv2.imshow('hsv', hsv)
    if cv2.waitKey(100) & 0xFF == 27:
        print '中止播放'
        break

cap.release()
cv2.destroyAllWindows()
