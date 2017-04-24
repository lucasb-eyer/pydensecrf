import numpy as np
import pydensecrf.densecrf as dcrf

# TODO: Make this real unit-tests some time in the future...

# Tests for specific issues
###########################

# Via e-mail: crash when non-float32 compat
d = dcrf.DenseCRF2D(10,10,2)
d.setUnaryEnergy(np.ones((2,10*10), dtype=np.float32))
compat = np.array([1.0, 2.0])
try:
    d.addPairwiseBilateral(sxy=(3,3), srgb=(3,3,3), rgbim=np.zeros((10,10,3), np.uint8), compat=compat)
    d.inference(2)
    raise TypeError("Didn't raise an exception, but should because compat dtypes don't match!!")
except ValueError:
    pass  # That's what we want!


# The following is not a really good unittest, but was the first tests.
###########################

# d = densecrf.PyDenseCRF2D(3, 2, 3)
# U = np.full((3,6), 0.1, dtype=np.float32)
# U[0,0] = U[1,1] = U[2,2] = U[0,3] = U[1,4] = U[2,5] = 0.8

d = dcrf.DenseCRF2D(10, 10, 2)

U1 = np.zeros((10, 10), dtype=np.float32)
U1[:,[0,-1]] = U1[[0,-1],:] = 1

U2 = np.zeros((10, 10), dtype=np.float32)
U2[4:7,4:7] = 1

U = np.vstack([U1.flat, U2.flat])
Up = (U + 1) / (np.sum(U, axis=0) + 2)

img = np.zeros((10,10,3), dtype=np.uint8)
img[2:8,2:8,:] = 255

d.setUnaryEnergy(-np.log(Up))
#d.setUnaryEnergy(PyConstUnary(-np.log(Up)))

d.addPairwiseBilateral(2, 2, img, 3)
# d.addPairwiseBilateral(2, 2, img, 3)
np.argmax(d.inference(10), axis=0).reshape(10,10)

