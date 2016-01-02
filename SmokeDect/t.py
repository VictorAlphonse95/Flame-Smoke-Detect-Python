# *^_^* coding:utf-8 *^_^*
__author__ = 'stone'
__date__ = '15-11-10'

import cv2
import numpy as np

vName = "../../videos/CTC_FG.028_9.mpg"

cap = cv2.VideoCapture(vName)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

while True:
    ret, frame = cap.read()
    height, width, ret = frame.shape
    n = 3
    small_frame = cv2.resize(frame, (width / n, height / n), interpolation=cv2.INTER_CUBIC)
    fmask = fgbg.apply(small_frame)

    opening_kernal = np.ones((3, 3), np.uint8)
    fmask = cv2.morphologyEx(fmask, cv2.MORPH_OPEN, opening_kernal)
    fmask = cv2.morphologyEx(fmask, cv2.MORPH_CLOSE, opening_kernal)
    cv2.imshow('fmask', fmask)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
