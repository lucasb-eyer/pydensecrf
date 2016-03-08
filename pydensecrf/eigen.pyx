# distutils: language = c++
# distutils: sources = pydensecrf/eigen_impl.cpp
# distutils: include_dirs = pydensecrf/densecrf/include


# [::1] means we want a C-contiguous array.
cdef c_VectorXf c_vectorXf(float[::1] arr):
    return c_buf2vecf(&arr[0], arr.shape[0])


def vectorXf(float[::1] arr not None):
    return VectorXf().wrap(c_vectorXf(arr))


cdef class VectorXf:

    def __cinit__(self):
        # Can't directly initialize v on construction because arguments
        # passed to `__cinit__` need to be Python objects. Refs:
        # - https://mail.python.org/pipermail/cython-devel/2012-June/002734.html
        # - https://kaushikghose.wordpress.com/2015/03/08/cython-__cinit__/
        self.shape = (0,)
        self.strides = (0,)

    cdef VectorXf wrap(self, c_VectorXf v):
        self.v = v
        self.shape[0] = v.size()
        self.strides[0] = sizeof(float)
        return self

    # http://docs.cython.org/src/userguide/buffer.html
    def __getbuffer__(self, Py_buffer *buf, int flags):
        buf.buf = <char *>self.v.data()
        buf.format = 'f'
        buf.internal = NULL
        buf.itemsize = sizeof(float)
        buf.len = self.shape[0] * buf.itemsize
        buf.ndim = 1
        buf.obj = self
        buf.readonly = 0
        buf.shape = self.shape
        buf.strides = self.strides
        buf.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buf):
        pass


# [:,::1] means we want a C-contiguous array.
cdef c_MatrixXf c_matrixXf(float[:,::1] arr):
    return c_buf2matf(&arr[0,0], arr.shape[0], arr.shape[1])


def matrixXf(float[:,::1] arr not None):
    return MatrixXf().wrap(c_matrixXf(arr))


cdef class MatrixXf:

    def __cinit__(self):
        # Can't directly initialize m on construction because arguments
        # passed to `__cinit__` need to be Python objects. Refs:
        # - https://mail.python.org/pipermail/cython-devel/2012-June/002734.html
        # - https://kaushikghose.wordpress.com/2015/03/08/cython-__cinit__/
        self.shape = (0,0)
        self.strides = (0,0)

    cdef MatrixXf wrap(self, c_MatrixXf m):
        self.m = m
        self.shape = (m.rows(), m.cols())

        # From http://docs.cython.org/src/userguide/buffer.html:
        # > Stride 1 is the distance, in bytes, between two items in a row;
        # > this is the distance between two adjacent items in the vector.
        # > Stride 0 is the distance between the first elements of adjacent rows.
        #
        # Since eigen's matrix (MatrixXf) is col-major, we've got:
        self.strides[0] = sizeof(float)
        self.strides[1] = self.shape[0] * self.strides[0]

        return self

    # http://docs.cython.org/src/userguide/buffer.html
    def __getbuffer__(self, Py_buffer *buf, int flags):
        buf.buf = <char *>self.m.data()
        buf.format = 'f'
        buf.internal = NULL
        buf.itemsize = sizeof(float)
        buf.len = self.shape[0] * self.shape[1] * buf.itemsize
        buf.ndim = 2
        buf.obj = self
        buf.readonly = 0
        buf.shape = self.shape
        buf.strides = self.strides
        buf.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buf):
        pass
