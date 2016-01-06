# *^_^* coding:utf-8 *^_^*
"""
SmokeDect-v0.2
"""

__author__ = 'stone'
__date__ = '15-11-30'

import cv2
import numpy as np
from numpy.linalg import norm

vName = '../../videos/CTC_FG.028_9.mpg'
# CTC_FG.028_9.mpg
# Homewood_BGsmokey.050_10.mpg
# Heavenly_FG.052_09.mpg
# CTC.BG.055_11.mpg
# camera2.mov
# 3_2012-07-17_15-15-44.avi


def hog_feature(img):
    """
    Extract HOG feature
    :param path:
    :return:
    """
    padding = (16, 9)
    block_stride = (16, 9)
    descriptor = cv2.HOGDescriptor()
    hog = descriptor.compute(img, block_stride, padding)  # #############运行到这里出错 原因应该是hog参数问题########
    return hog


def split(img, cell_size, flatten=True):
    """
    Split into small blocks
    :param img:
    :param cell_size:
    :param flatten:
    :return:
    """
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w // sx) for row in np.vsplit(img, h // sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells


def zoom_down(frames, n):
    """
    zoom down/up the image
    """
    h, w, r = frames.shape  # h:height w:width r:ret
    small_frames = cv2.resize(frame, (w / n, h / n), interpolation=cv2.INTER_CUBIC)
    return small_frames


if __name__ == '__main__':
    print __doc__

    cap = cv2.VideoCapture(vName)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    svm = cv2.ml.SVM_create()
    svm.setGamma(0.5)
    svm.setC(1)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setType(cv2.ml.SVM_C_SVC)
    # svm.train(train_hog, cv2.ml.ROW_SAMPLE, response)

    while True:
        ret, frame = cap.read()
        if frame is None:
            print "Video playback is completed"
            break

        small_frame = zoom_down(frame, 3)
        gray_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        fmask = fgbg.apply(gray_small_frame)

        open_kernel = np.ones((5, 5), np.uint8)
        # close_kernel = np.ones((32, 32), np.uint8)
        fmask_open = cv2.morphologyEx(fmask, cv2.MORPH_OPEN, open_kernel)
        # fmask_close = cv2.morphologyEx(fmask_open, cv2.MORPH_CLOSE, close_kernel)

        small_frame_and = cv2.bitwise_and(small_frame, small_frame, mask=fmask_open)
        small_frame_and = cv2.dilate(small_frame_and, np.ones((32, 32), np.uint8))

        # cv2.imshow('fmask', fmask_close)
        cv2.imshow("frame", small_frame_and)
        # 提取待检测区域
        # 帧差法提取运动区域
        # 根据模糊度或者颜色或者运动方向精确确认待检测区域
        # 提取待检测区域特征，包括hog，lbp，边缘，运动能量，小波分析等特征
        # 用svm进行训练和识别

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
