#! /usr/bin/env python
# coding=utf-8
import numpy as np
import cv2

img = cv2.imread('../lena.jpg')

cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
