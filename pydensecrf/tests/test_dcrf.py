import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

import pytest


def _get_simple_unary():
    unary1 = np.zeros((10, 10), dtype=np.float32)
    unary1[:, [0, -1]] = unary1[[0, -1], :] = 1

    unary2 = np.zeros((10, 10), dtype=np.float32)
    unary2[4:7, 4:7] = 1

    unary = np.vstack([unary1.flat, unary2.flat])
    unary = (unary + 1) / (np.sum(unary, axis=0) + 2)

    return unary


def _get_simple_img():

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[2:8, 2:8, :] = 255

    return img


def test_call_dcrf2d():

    d = dcrf.DenseCRF2D(10, 10, 2)

    unary = _get_simple_unary()
    img = _get_simple_img()

    d.setUnaryEnergy(-np.log(unary))
    # d.setUnaryEnergy(PyConstUnary(-np.log(Up)))

    d.addPairwiseBilateral(sxy=2, srgb=2, rgbim=img, compat=3)
    # d.addPairwiseBilateral(2, 2, img, 3)
    np.argmax(d.inference(10), axis=0).reshape(10, 10)


def test_call_dcrf():

    d = dcrf.DenseCRF(100, 2)

    unary = _get_simple_unary()
    img = _get_simple_img()

    d.setUnaryEnergy(-np.log(unary))
    # d.setUnaryEnergy(PyConstUnary(-np.log(Up)))

    feats = utils.create_pairwise_bilateral(sdims=(2, 2), schan=2,
                                            img=img, chdim=2)

    d.addPairwiseEnergy(feats, compat=3)
    # d.addPairwiseBilateral(2, 2, img, 3)
    np.argmax(d.inference(10), axis=0).reshape(10, 10)


def test_call_dcrf_eq_dcrf2d():

    d = dcrf.DenseCRF(100, 2)
    d2 = dcrf.DenseCRF2D(10, 10, 2)

    unary = _get_simple_unary()
    img = _get_simple_img()

    d.setUnaryEnergy(-np.log(unary))
    d2.setUnaryEnergy(-np.log(unary))

    feats = utils.create_pairwise_bilateral(sdims=(2, 2), schan=2,
                                            img=img, chdim=2)

    d.addPairwiseEnergy(feats, compat=3)

    d2.addPairwiseBilateral(sxy=2, srgb=2, rgbim=img, compat=3)
    # d.addPairwiseBilateral(2, 2, img, 3)
    res1 = np.argmax(d.inference(10), axis=0).reshape(10, 10)
    res2 = np.argmax(d2.inference(10), axis=0).reshape(10, 10)

    assert(np.all(res1 == res2))


def test_compact_wrong():

    # Tests whether expection is indeed raised
    ##########################################

    # Via e-mail: crash when non-float32 compat
    d = dcrf.DenseCRF2D(10, 10, 2)
    d.setUnaryEnergy(np.ones((2, 10 * 10), dtype=np.float32))
    compat = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        d.addPairwiseBilateral(sxy=(3, 3), srgb=(3, 3, 3), rgbim=np.zeros(
            (10, 10, 3), np.uint8), compat=compat)
        d.inference(2)
