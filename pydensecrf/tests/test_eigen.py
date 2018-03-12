import numpy as np
import pydensecrf.eigen as e

import pytest


def test_vector_conversion():
    np_vector = np.random.randn(3).astype(np.float32)
    c_vector = e.vectorXf(np_vector)
    assert np.all(np.array(c_vector) == np_vector)


def test_matrix_conversion():
    np_matrix = np.random.randn(3, 3).astype(np.float32)
    assert(np_matrix.ndim == 2)
    c_matrix = e.matrixXf(np_matrix)
    assert np.all(np.array(c_matrix) == np_matrix)


def test_wrong_dims():
    np_matrix = np.random.randn(3, 3, 3).astype(np.float32)
    assert(np_matrix.ndim == 3)
    # c_matrix only supports ndim == 2
    with pytest.raises(ValueError):
        # Check whether a Value Error is raised
        e.matrixXf(np_matrix)


def test_wrong_type():
    np_matrix = np.random.randn(3, 3).astype(np.float64)
    # c_matrix requies type np.float32
    with pytest.raises(ValueError):
        # Check whether a Value Error is raised
        e.matrixXf(np_matrix)


def test_none_type():
    np_matrix = None
    with pytest.raises(TypeError):
        # Check whether a Value Error is raised
        e.matrixXf(np_matrix)
