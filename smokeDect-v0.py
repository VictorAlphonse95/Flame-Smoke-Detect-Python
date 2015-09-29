# *_* coding:utf-8 *_*

__author__ = 'stone'
__date__ = '15-9-28'

import numpy as np
import cv2

vName = 'videos/CTC_FG.028_9.mpg'
# CarLights1.avi
# CTC_FG.028_9.mpg
# SmokeVideo.mp4

cap = cv2.VideoCapture(vName)

fgbg = cv2.createBackgroundSubtractorKNN()

while True:
    ret, frame = cap.read()
    if frame is None:
        print("视频读取完毕！")
        break
    height, width, ret = frame.shape
    # 视频旋转问题
    # center = (height, width)
    # angle = 30
    # scale = 1
    # rotate = cv2.getRotationMatrix2D(center, angle, scale)

    # 缩小 n 倍显示视频
    n = 3
    small_frame = cv2.resize(frame, (width / n, height / n), interpolation=cv2.INTER_CUBIC)

    fgmask = fgbg.apply(small_frame)

    GsBlur = cv2.GaussianBlur(fgmask, (5, 5), 0)
    # kernel = np.ones((5, 5), np.uint8)
    # opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow("video", small_frame)
    cv2.imshow("frame", fgmask)
    cv2.imshow("test", GsBlur)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        print("终止播放！")
        break

cap.release()
cv2.destroyAllWindows()
