# *-*coding:utf-8*-*

__author__ = 'stone'
__date__ = '15-10-23'

import cv2

vName = '../images/basketball1.png'
# CTC_FG.028_9.mpg
# Homewood_BGsmokey.050_10.mpg
# Heavenly_FG.052_09.mpg
# CTC.BG.055_11.mpg

# cap = cv2.VideoCapture(vName)

img = cv2.imread(vName)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())

found, w = hog.detectMultiScale(img, winStride=(8, 8), padding=(32, 32), scale=1.05)

print found
print w
