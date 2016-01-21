# *^_^* coding:utf-8 *^_^*

from __future__ import print_function
import cv2
import numpy as np
import os

train_directory = '/home/stone/Documents/HogFeature'
test_directory = '/home/stone/Documents/test_svm'


def file_path(directory):
    files = os.listdir(directory)
    path = []

    for name in files:
        full_name = os.path.join(directory, name)
        path.append(full_name)
    print('%s 的文件数: %d\n' % (directory, len(path)))
    return path


def load_hog(hog_path):
    print('loading hog files...')
    hog = []
    for fp in hog_path:
        content = np.loadtxt(fp)
        hog.append(content)
    hog = np.float32(hog)
    print('finished!')
    return hog


def svm():
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

response = [1]*3768
response = np.array(response)
print('count of response: %d' % len(response))
svm = svm()

train_path = file_path(train_directory)
test_path = file_path(test_directory)

train_hog = load_hog(train_path)

for fp in test_path:
    test_hog = np.loadtxt(fp)
    test_hog = np.float32(test_hog)
    svm.train(train_hog, cv2.ml.ROW_SAMPLE, response)
    result = svm.predict(test_hog)
    print(result)
