"""
Usage: python util_inference_example.py image annotations

Adapted from the dense_inference.py to demonstate the usage of the util
functions.
"""

import sys
import numpy as np
import cv2
import densecrf as dcrf
import matplotlib.pylab as plt
from skimage.segmentation import relabel_sequential

from utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian

fn_im = sys.argv[1]
fn_anno = sys.argv[2]

##################################
### Read images and annotation ###
##################################
img = cv2.imread(fn_im)
labels = relabel_sequential(cv2.imread(fn_anno, 0))[0].flatten()
M = 21 # 21 Classes to match the C++ example

###########################
### Setup the CRF model ###
###########################
use_2d = False
if use_2d:
    # Example using the DenseCRF2D code
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    # get unary potentials (neg log probability)
    U = compute_unary(labels, M)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
else:
    # Example using the DenseCRF class and the util functions
    d = dcrf.DenseCRF(img.shape[0] * img.shape[1], M)

    # get unary potentials (neg log probability)
    U = compute_unary(labels, M)
    d.setUnaryEnergy(U)

    # This creates the color-independent features and then add them to the CRF
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                      img=img, chdim=2)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)


####################################
### Do inference and compute map ###
####################################
Q = d.inference(5)
map = np.argmax(Q, axis=0).reshape(img.shape[:2])

res = map.astype('float32') * 255 / map.max()
plt.imshow(res)
plt.show()


# Manually inference
Q, tmp1, tmp2 = d.startInference()
for i in range(5):
    print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
    d.stepInference(Q, tmp1, tmp2)

