# *^_^* coding:utf-8 *^_^*
"""
把视频统一输出为某一尺寸，或者原视频的n倍.
也可以截取视频中的某一区域。
保存格式为avi
"""

from __future__ import print_function

__author__ = 'stone'
__date__ = '16-1-15'

import cv2
import numpy as np

ZOOM_TIME = 3
FRAME_SIZE = (320, 240)  # (WIDTH, HEIGHT)
VIDEO_SRC = ''
VIDEO_SAVE_PATH = ''


def zoom_down(frames, time=None, size=None):
    """
    zoom down/up the image
    time:视频缩小的倍数
    size:要输出的视频的尺寸，size（width, height）
    """
    if time is not None:
        h, w, r = frames.shape  # h:height w:width r:ret
        small_frames = cv2.resize(frame, (w / time, h / time), interpolation=cv2.INTER_CUBIC)
    elif size is not None:
        small_frames = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
    return small_frames


if __name__ == '__main__':
    cap = cv2.VideoCapture(VIDEO_SRC)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(VIDEO_SAVE_PATH, fourcc, 20.0, FRAME_SIZE)

    while True:
        new_frame = np.zeros((240, 320, 3), np.uint8)
        ret, frame = cap.read()
        if ret:
            frame = zoom_down(frame, size=FRAME_SIZE)
            # new_frame[:, :] = frame[600:840, 1100:1420]  # 用于截取某一区域

            out.write(frame)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            print('The End')
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
