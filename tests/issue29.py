# probs of shape 3d image per class: Nb_classes x Height x Width x Depth
# assume the image has shape (69, 51, 72)
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian

###

#shape = (69, 51, 72)
#probs = np.random.randn(5, 69, 51).astype(np.float32)
#probs /= probs.sum(axis=0, keepdims=True)
#
#d = dcrf.DenseCRF(np.prod(shape), probs.shape[0])
#U = unary_from_softmax(probs)
#print(U.shape)
#d.setUnaryEnergy(U)
#feats = create_pairwise_gaussian(sdims=(1.0, 1.0, 1.0), shape=shape)
#d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.FULL_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
#Q = d.inference(5)
#new_image = np.argmax(Q, axis=0).reshape((shape[0], shape[1],shape[2]))


###

SHAPE, NLABELS = (69, 51, 72), 5
probs = np.random.randn(NLABELS, 68, 50).astype(np.float32)  # WRONG shape here
probs /= probs.sum(axis=0, keepdims=True)

d = dcrf.DenseCRF(np.prod(SHAPE), NLABELS)

d.setUnaryEnergy(unary_from_softmax(probs))  # THIS SHOULD THROW and not crash later
feats = create_pairwise_gaussian(sdims=(1.0, 1.0, 1.0), shape=SHAPE)
d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.FULL_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

Q = d.inference(5)
new_image = np.argmax(Q, axis=0).reshape(SHAPE)
