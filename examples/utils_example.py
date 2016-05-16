"""
Adapted from the inference.py to demonstate the usage of the util functions.
"""

import sys
import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from skimage.segmentation import relabel_sequential

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian

if len(sys.argv) != 4:
    print("Usage: python {} IMAGE ANNO OUTPUT".format(sys.argv[0]))
    print("")
    print("IMAGE and ANNO are inputs and OUTPUT is where the result should be written.")
    sys.exit(1)

fn_im = sys.argv[1]
fn_anno = sys.argv[2]
fn_output = sys.argv[3]

##################################
### Read images and annotation ###
##################################
img = cv2.imread(fn_im)
labels, _, _ = relabel_sequential(cv2.imread(fn_anno, 0))

# Compute the number of classes in the label image
M = len(set(labels.flat))

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
MAP = np.argmax(Q, axis=0).astype('float32')
MAP *= 255 / MAP.max()
MAP = MAP.reshape(img.shape[:2])
cv2.imwrite(fn_output, MAP.astype('uint8'))

# Manually inference
Q, tmp1, tmp2 = d.startInference()
for i in range(5):
    print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
    d.stepInference(Q, tmp1, tmp2)
