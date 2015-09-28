# *_* coding:utf-8 *_*

__author__ = 'stone'
__date__ = '15-9-28'

import numpy as np
import cv2

vName = 'videos/CTC_FG.028_9.mpg'
# CarLights1.avi
# CTC_FG.028_9.mpg

cap = cv2.VideoCapture(vName)

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    medianBlur = cv2.medianBlur(fgmask, 5)
    if frame is None:
        print("视频读取完毕！")
        break

    # cv2.imshow("frame", frame)
    cv2.imshow("fgbg", medianBlur)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        print("终止播放！")
        break

cap.release()
cv2.destroyAllWindows()
