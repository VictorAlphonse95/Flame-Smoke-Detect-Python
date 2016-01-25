# *^_^* coding:utf-8 *^_^*

from __future__ import print_function
import cv2
import numpy as np
import os


img = cv2.imread('../medias/lena.jpg')
h, w = img.shape[:2]
print(h, w)
sx, sy = (32, 32)
cells = [np.hsplit(row, w // sx) for row in np.vsplit(img, h // sy)]
cells = np.array(cells)
print(cells[0][0][0])
print(cells.shape)
# cells = cells.reshape(-1, sy, sx)
