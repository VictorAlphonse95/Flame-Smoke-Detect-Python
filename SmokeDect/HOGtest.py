# *-*coding:utf-8*-*

__author__ = 'stone'
__date__ = '15-10-23'

import cv2

vName = '../videos/CTC_FG.028_9.mpg'
# CTC_FG.028_9.mpg
# Homewood_BGsmokey.050_10.mpg
# Heavenly_FG.052_09.mpg
# CTC.BG.055_11.mpg

cap = cv2.VideoCapture(vName)


def draw_detections(img, rects, thickness=1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15 * w), int(0.05 * h)
        cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 0, 255), thickness)


hog = cv2.HOGDescriptor()
# hog.save('hog.xml')  # 将hog参数保存到xml
# hog = cv2.HOGDescriptor('hog.xml')  # 从xml中读取参数
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
while True:
    ret, frame = cap.read()
    if frame is None:
        print("视频读取完毕！")
        break

    found, w = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)

    print found
    print w

    cv2.imshow('img', frame)

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break
cv2.destroyAllWindows()
