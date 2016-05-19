import numpy as np


def compute_unary(labels, M, GT_PROB=0.5):
    """
    Simple classifier that is 50% certain that the annotation is correct.
    (same as in the inference example).


    Parameters
    ----------
    labels: nummpy.array
        The label-map. The label value `0` is not a label, but the special
        value indicating that the location has no label/information and thus
        every label is equally likely.
    M: int
        The number of labels there are, not including the special `0` value.
    GT_PROB: float
        The certainty of the ground-truth (must be within (0,1)).
    """
    assert 0 < GT_PROB < 1, "`GT_PROB must be in (0,1)."

    labels = labels.flatten()

    u_energy = -np.log(1.0 / M)
    n_energy = -np.log((1.0 - GT_PROB) / (M - 1))
    p_energy = -np.log(GT_PROB)

    U = np.zeros((M, len(labels)), dtype='float32')
    U[:, labels > 0] = n_energy
    U[labels, np.arange(U.shape[1])] = p_energy
    U[:, labels == 0] = u_energy
    return U


def softmax_to_unary(sm, GT_PROB=1):
    """
    Util function that converts softmax scores (classwise probabilities) to
    unary potentials (the negative log likelihood per node).

    Parameters
    ----------
    sm: nummpy.array
        Softmax input. The first dimension is expected to be the classes,
        all others will be flattend.
    GT_PROB: float
        The certainty of the softmax output (default is 1).

    """
    num_cls = sm.shape[0]
    if GT_PROB < 1:
        uniform = np.ones(sm.shape) / num_cls
        sm = GT_PROB * sm + (1 - GT_PROB) * uniform
    return -np.log(sm).reshape([num_cls, -1]).astype(np.float32)


def create_pairwise_gaussian(sdims, shape):
    """
    Util function that create pairwise gaussian potentials. This works for all
    image dimensions. For the 2D case does the same as
    `DenseCRF2D.addPairwiseGaussian`.

    Parameters
    ----------
    sdims: list or tuple
        The scaling factors per dimension. This is referred to `sxy` in
        `DenseCRF2D.addPairwiseGaussian`.
    shape: list or tuple
        The shape the CRF has.

    """
    # create mesh
    hcord_range = [range(s) for s in shape]
    mesh = np.array(np.meshgrid(*hcord_range, indexing='ij'), dtype=np.float32)

    # scale mesh accordingly
    for i, s in enumerate(sdims):
        mesh[i] /= s
    return mesh.reshape([len(sdims), -1])


def create_pairwise_bilateral(sdims, schan, img, chdim=-1):
    """
    Util function that create pairwise bilateral potentials. This works for
    all image dimensions. For the 2D case does the same as
    `DenseCRF2D.addPairwiseBilateral`.

    Parameters
    ----------
    sdims: list or tuple
        The scaling factors per dimension. This is referred to `sxy` in
        `DenseCRF2D.addPairwiseBilateral`.
    schan: list or tuple
        The scaling factors per channel in the image. This is referred to
        `srgb` in `DenseCRF2D.addPairwiseBilateral`.
    img: numpy.array
        The input image.
    chdim: int, optional
        This specifies where the channel dimension is in the image. For
        example `chdim=2` for a RGB image of size (240, 300, 3). If the
        image has no channel dimension (e.g. it has only one channel) use
        `chdim=-1`.

    """
    # Put channel dim in right position
    if chdim == -1:
        # We don't have a channel, add a new axis
        im_feat = img[np.newaxis].astype(np.float32)
    else:
        # Put the channel dim as axis 0, all others stay relatively the same
        im_feat = np.rollaxis(img, chdim).astype(np.float32)

    # scale image features per channel
    for i, s in enumerate(schan):
        im_feat[i] /= s

    # create a mesh
    cord_range = [range(s) for s in im_feat.shape[1:]]
    mesh = np.array(np.meshgrid(*cord_range, indexing='ij'), dtype=np.float32)

    # scale mesh accordingly
    for i, s in enumerate(sdims):
        mesh[i] /= s

    feats = np.concatenate([mesh, im_feat])
    return feats.reshape([feats.shape[0], -1])


def _create_pairwise_gaussian_2d(sx, sy, shape):
    """
    A simple reference implementation for the 2D case. The ND implementation
    is faster.
    """
    feat_size = 2
    feats = np.zeros((feat_size, shape[0], shape[1]), dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            feats[0, i, j] = i / sx
            feats[1, i, j] = j / sy
    return feats.reshape([feat_size, -1])


def _create_pairwise_bilateral_2d(sx, sy, sr, sg, sb, img):
    """
    A simple reference implementation for the 2D case. The ND implementation
    is faster.
    """
    feat_size = 5
    feats = np.zeros((feat_size, img.shape[0], img.shape[1]), dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            feats[0, i, j] = i / sx
            feats[1, i, j] = j / sy
            feats[2, i, j] = img[i, j, 0] / sr
            feats[3, i, j] = img[i, j, 1] / sg
            feats[4, i, j] = img[i, j, 2] / sb
    return feats.reshape([feat_size, -1])

