#!/usr/bin/python

"""
Usage: python dense_inference.py image annotations output

Adapted from the original C++ example: densecrf/examples/dense_inference.cpp
http://www.philkr.net/home/densecrf Version 2.2
"""

import numpy as np
import cv2
import densecrf as dcrf
from skimage.segmentation import relabel_sequential
import sys

img = cv2.imread(sys.argv[1], 1)
labels = relabel_sequential(cv2.imread(sys.argv[2], 0))[0].flatten()
output = sys.argv[3]

M = labels.max() + 1  # number of labels

# Setup the CRF model
d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

# Certainty that the ground truth is correct
GT_PROB = 0.5

# Simple classifier that is 50% certain that the annotation is correct
u_energy = -np.log(1.0 / M)
n_energy = -np.log((1.0 - GT_PROB) / (M - 1))
p_energy = -np.log(GT_PROB)

U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
U[:, labels > 0] = n_energy
U[labels, np.arange(U.shape[1])] = p_energy
U[:, labels == 0] = u_energy
d.setUnaryEnergy(U)

d.addPairwiseGaussian(sxy=3, compat=3)
d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)

# Do the inference
res = np.argmax(d.inference(5), axis=0).astype('float32')

res *= 255 / res.max()
res = res.reshape(img.shape[:2])
cv2.imwrite(output, res.astype('uint8'))
