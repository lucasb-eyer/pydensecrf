# distutils: language = c++
# distutils: sources = pydensecrf/densecrf/src/densecrf.cpp pydensecrf/densecrf/src/unary.cpp pydensecrf/densecrf/src/pairwise.cpp pydensecrf/densecrf/src/permutohedral.cpp pydensecrf/densecrf/src/optimization.cpp pydensecrf/densecrf/src/objective.cpp pydensecrf/densecrf/src/labelcompatibility.cpp pydensecrf/densecrf/src/util.cpp pydensecrf/densecrf/external/liblbfgs/lib/lbfgs.c
# distutils: include_dirs = pydensecrf/densecrf/include pydensecrf/densecrf/external/liblbfgs/include

from numbers import Number

import eigen
cimport eigen


cdef LabelCompatibility* _labelcomp(compat):
    if isinstance(compat, Number):
        return new PottsCompatibility(compat)
    elif memoryview(compat).ndim == 1:
        return new DiagonalCompatibility(eigen.c_vectorXf(compat))
    elif memoryview(compat).ndim == 2:
        return new MatrixCompatibility(eigen.c_matrixXf(compat))
    else:
        raise ValueError("LabelCompatibility of dimension >2 not meaningful.")


cdef class Unary:

    # Because all of the APIs that take an object of this type will
    # take ownership. Thus, we need to make sure not to delete this
    # upon destruction.
    cdef UnaryEnergy* move(self):
        ptr = self.thisptr
        self.thisptr = NULL
        return ptr

    # It might already be deleted by the library, actually.
    # Yeah, pretty sure it is.
    def __dealloc__(self):
        del self.thisptr


cdef class ConstUnary(Unary):
    def __cinit__(self, float[:,::1] u not None):
        self.thisptr = new ConstUnaryEnergy(eigen.c_matrixXf(u))


cdef class LogisticUnary(Unary):
    def __cinit__(self, float[:,::1] L not None, float[:,::1] f not None):
        self.thisptr = new LogisticUnaryEnergy(eigen.c_matrixXf(L), eigen.c_matrixXf(f))


cdef class DenseCRF:

    def __cinit__(self, int nvar, int nlabels, *_, **__):
        # We need to swallow extra-arguments because superclass cinit function
        # will always be called with the same params as the subclass, automatically.

        # We also only want to avoid creating an object if we're just being called
        # from a subclass as part of the hierarchy.
        if type(self) is DenseCRF:
            self._this = new c_DenseCRF(nvar, nlabels)
        else:
            self._this = NULL

    def __dealloc__(self):
        # Because destructors are virtual, this is enough to delete any object
        # of child classes too.
        if self._this:
            del self._this

    def addPairwiseEnergy(self, float[:,::1] features not None, compat, KernelType kernel=DIAG_KERNEL, NormalizationType normalization=NORMALIZE_SYMMETRIC):
        self._this.addPairwiseEnergy(eigen.c_matrixXf(features), _labelcomp(compat), kernel, normalization)

    def setUnary(self, Unary u):
        self._this.setUnaryEnergy(u.move())

    def setUnaryEnergy(self, float[:,::1] u not None, float[:,::1] f = None):
        if f is None:
            self._this.setUnaryEnergy(eigen.c_matrixXf(u))
        else:
            self._this.setUnaryEnergy(eigen.c_matrixXf(u), eigen.c_matrixXf(f))

    def inference(self, int niter):
        return eigen.MatrixXf().wrap(self._this.inference(niter))

    def startInference(self):
        return eigen.MatrixXf().wrap(self._this.startInference()), eigen.MatrixXf(), eigen.MatrixXf()

    def stepInference(self, MatrixXf Q, MatrixXf tmp1, MatrixXf tmp2):
        self._this.stepInference(Q.m, tmp1.m, tmp2.m)

    def klDivergence(self, MatrixXf Q):
        return self._this.klDivergence(Q.m)


cdef class DenseCRF2D(DenseCRF):

    # The same comments as in the superclass' `__cinit__` apply here.
    def __cinit__(self, int w, int h, int nlabels, *_, **__):
        if type(self) is DenseCRF2D:
            self._this = self._this2d = new c_DenseCRF2D(w, h, nlabels)

    def addPairwiseGaussian(self, sxy, compat, KernelType kernel=DIAG_KERNEL, NormalizationType normalization=NORMALIZE_SYMMETRIC):
        if isinstance(sxy, Number):
            sxy = (sxy, sxy)

        self._this2d.addPairwiseGaussian(sxy[0], sxy[1], _labelcomp(compat), kernel, normalization)

    def addPairwiseBilateral(self, sxy, srgb, unsigned char[:,:,::1] rgbim not None, compat, KernelType kernel=DIAG_KERNEL, NormalizationType normalization=NORMALIZE_SYMMETRIC):
        if isinstance(sxy, Number):
            sxy = (sxy, sxy)

        if isinstance(srgb, Number):
            srgb = (srgb, srgb, srgb)

        self._this2d.addPairwiseBilateral(
            sxy[0], sxy[1], srgb[0], srgb[1], srgb[2], &rgbim[0,0,0], _labelcomp(compat), kernel, normalization
        )
