# *^_^* coding:utf-8 *^_^*
__author__ = 'stone'
__date__ = '16-1-5'

import numpy as np
import cv2

vName = '../../videos/side/daria_side.avi'
# CTC_FG.028_9.mpg
# Homewood_BGsmokey.050_10.mpg
# Heavenly_FG.052_09.mpg
# camera2.mov
# 3_2012-07-17_15-15-44.avi
# ../../videos/side/daria_side.avi


def zoom_down(frame, n):
    """
    zoom down/up the image
    """
    h, w, r = frame.shape  # h:height w:width r:ret
    small_frames = cv2.resize(frame, (w / n, h / n), interpolation=cv2.INTER_CUBIC)
    return small_frames


if __name__ == '__main__':

    cap = cv2.VideoCapture(vName)

    feature_params = dict(
        maxCorners=300,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=200
    )

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    color = np.random.randint(0, 255, (255, 3))

    ret, old_frame = cap.read()
    # old_frame = zoom_down(old_frame, 3)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    mask = np.zeros_like(old_frame)
    count = 1

    while True:
        ret, frame = cap.read()
        # frame = zoom_down(frame, 3)
        if frame is None:
            print "Video playback is completed"
            break

        # if (count % 25) == 0:
        #     old_frame = frame
        #     old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        #     p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        #
        #     mask = np.zeros_like(old_frame)
        #     count += 1
        # else:
        #     count += 1

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 3, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(200) & 0xFF
        if k == 27:
            break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    cap.release()
