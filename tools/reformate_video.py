# *^_^* coding:utf-8 *^_^*
"""
把视频统一输出为某一尺寸，或者原视频的n倍
"""
__author__ = 'stone'
__date__ = '16-1-15'

import cv2

ZOOM_TIME = 3
FRAME_SIZE = (320, 240)  # (WIDTH, HEIGHT)
VIDEO_SRC = '/home/stone/Code/FlameSmokeDetect/medias/videos/CTC_FG.028_9.mpg'
VIDEO_SAVE_PATH = 'new.avi'


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
        ret, frame = cap.read()
        if ret:
            frame = zoom_down(frame, size=FRAME_SIZE)

            out.write(frame)

            # cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            print 'The End'
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
