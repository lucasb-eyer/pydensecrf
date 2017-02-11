import numpy as np
from numbers import Number
from logging import warning


def unary_from_labels(labels, n_labels, gt_prob, zero_unsure=True):
    """
    Simple classifier that is 50% certain that the annotation is correct.
    (same as in the inference example).


    Parameters
    ----------
    labels: numpy.array
        The label-map, i.e. an array of your data's shape where each unique
        value corresponds to a label.
    n_labels: int
        The total number of labels there are.
        If `zero_unsure` is True (the default), this number should not include
        `0` in counting the labels, since `0` is not a label!
    gt_prob: float
        The certainty of the ground-truth (must be within (0,1)).
    zero_unsure: bool
        If `True`, treat the label value `0` as meaning "could be anything",
        i.e. entries with this value will get uniform unary probability.
        If `False`, do not treat the value `0` specially, but just as any
        other class.
    """
    assert 0 < gt_prob < 1, "`gt_prob must be in (0,1)."

    labels = labels.flatten()

    n_energy = -np.log((1.0 - gt_prob) / (n_labels - 1))
    p_energy = -np.log(gt_prob)

    # Note that the order of the following operations is important.
    # That's because the later ones overwrite part of the former ones, and only
    # after all of them is `U` correct!
    U = np.full((n_labels, len(labels)), n_energy, dtype='float32')
    U[labels - 1 if zero_unsure else labels, np.arange(U.shape[1])] = p_energy

    # Overwrite 0-labels using uniform probability, i.e. "unsure".
    if zero_unsure:
        U[:, labels == 0] = -np.log(1.0 / n_labels)

    return U


def compute_unary(labels, M, GT_PROB=0.5):
    """Deprecated, use `unary_from_labels` instead."""
    warning("pydensecrf.compute_unary is deprecated, use unary_from_labels instead.")
    return unary_from_labels(labels, M, GT_PROB)


def unary_from_softmax(sm, scale=None, clip=1e-5):
    """Converts softmax class-probabilities to unary potentials (NLL per node).

    Parameters
    ----------
    sm: numpy.array
        Output of a softmax where the first dimension is the classes,
        all others will be flattend. This means `sm.shape[0] == n_classes`.
    scale: float
        The certainty of the softmax output (default is None).
        If not None, the softmax outputs are scaled to range from uniform
        probability for 0 outputs to `scale` probability for 1 outputs.
    clip: float
        Minimum value to which probability should be clipped.
        This is because the unary is the negative log of the probability, and
        log(0) = inf, so we need to clip 0 probabilities to a positive value.
    """
    num_cls = sm.shape[0]
    if scale is not None:
        assert 0 < scale <= 1, "`scale` needs to be in (0,1]"
        uniform = np.ones(sm.shape) / num_cls
        sm = scale * sm + (1 - scale) * uniform
    if clip is not None:
        sm = np.clip(sm, clip, 1.0)
    return -np.log(sm).reshape([num_cls, -1]).astype(np.float32)


def softmax_to_unary(sm, GT_PROB=1):
    """Deprecated, use `unary_from_softmax` instead."""
    warning("pydensecrf.softmax_to_unary is deprecated, use unary_from_softmax instead.")
    scale = None if GT_PROB == 1 else GT_PROB
    return unary_from_softmax(sm, scale, clip=None)


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
    # Allow for a single number in `schan` to broadcast across all channels:
    if isinstance(schan, Number):
        im_feat /= schan
    else:
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

