# *^_^* coding:utf-8 *^_^*
"""
SmokeDect-v0.2
"""

from __future__ import print_function  # use print()

__author__ = 'stone'
__date__ = '15-11-30'

import cv2
import numpy as np
from time import time
import os

DEBUG_MOD = True
vName = '../medias/videos/CTC_FG.028_9_320x240.avi'
train_directory = '/home/stone/Documents/hog'
BLOCK_SIZE = 30


def hog_feature(img):
    """
    Extract HOG feature
    """
    win_size = (16, 16)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    descriptor = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog = descriptor.compute(img)
    return hog


def svm_create():
    """
    initial a SVM
    """
    svm_params = dict(kernel_type=cv2.ml.SVM_LINEAR,
                      svm_type=cv2.ml.SVM_C_SVC,
                      C=2.67,
                      gamma=5.385)
    svm = cv2.ml.SVM_create()
    svm.setKernel(svm_params['kernel_type'])
    svm.setType(svm_params['svm_type'])
    svm.setC(svm_params['C'])
    svm.setGamma(svm_params['gamma'])
    return svm


def file_path(directory):
    """
    generate a full directory path
    """
    files = os.listdir(directory)
    path = []

    for name in files:
        full_name = os.path.join(directory, name)
        path.append(full_name)
    print('%s 的文件数: %d\n' % (directory, len(path)))
    return path


def load_hog(hog_path):
    """
    load hog files
    """
    print('loading hog files...')
    hog = []
    for fp in hog_path:
        content = np.loadtxt(fp)
        hog.append(content)
    hog = np.float32(hog)
    print('finished!')
    return hog


def split(img, cell_size, flatten=True):
    """
    Split into small blocks
    """
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w // sx) for row in np.vsplit(img, h // sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells


if __name__ == '__main__':
    print(__doc__)
    start_time = time()
    count = 0

    # create and train a SVM
    train_path = file_path(train_directory)
    train_hog = load_hog(train_path)
    response = [1]*760
    response = np.array(response)
    svm = svm_create()
    svm.train(train_hog, cv2.ml.ROW_SAMPLE, response)

    cap = cv2.VideoCapture(vName)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    while True:
        ret, frame = cap.read()
        if frame is None:
            print("Video playback is completed")
            break

        if count < 500:
            count += 1
            continue
        h, w = frame.shape[:2]
        frame_copy = frame.copy()
        frame = cv2.GaussianBlur(frame, (5, 5), 2)
        frame = cv2.medianBlur(frame, 5)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fmask = fgbg.apply(gray_frame)

        kernel = np.ones((20, 20), np.uint8)
        fmask = cv2.medianBlur(fmask, 3)
        fmask = cv2.dilate(fmask, kernel)

        fmask_copy = fmask.copy()

        contour_img, contours, hierarchy = cv2.findContours(fmask_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if w < BLOCK_SIZE or h < BLOCK_SIZE:
                continue
            else:
                cells = split(frame, (BLOCK_SIZE, BLOCK_SIZE), flatten=False)
                cells_x_1 = x / BLOCK_SIZE
                cells_x_2 = (x+w) / BLOCK_SIZE
                cells_y_1 = y / BLOCK_SIZE
                cells_y_2 = (y+h) / BLOCK_SIZE
                hog = []
                for j in xrange(cells_y_1, cells_y_2):
                    for i in xrange(cells_x_1, cells_x_2):
                        candidate = cells[j][i]
                        try:
                            hog.append(hog_feature(candidate))
                        except:
                            print('something wrong in line 103')

                        print('HogFeature/hog%d-%d-%d' % (count, i, j))
                        # hog_file = 'HogFeature/hog%d-%d-%d' % (count, i, j)
                        # np.savetxt(hog_file, hog)
                        hog = np.float32(hog)
                        result = svm.predict(hog)
                        print(result)

            if DEBUG_MOD is True:
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('fmask', frame_copy)
        frame_and = cv2.bitwise_and(frame, frame, mask=fmask)

        count += 1
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break

    end_time = time()
    total_time = end_time - start_time
    print('Total time: %d' % total_time)
    print(count)
    cap.release()
    cv2.destroyAllWindows()

