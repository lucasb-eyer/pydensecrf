import unittest

import numpy as np

import pydensecrf.utils as utils


class TestUnary(unittest.TestCase):

    def test_unary(self):
        M = 3
        U, P, N = 1./M, 0.8, 0.2/(M-1)  # Uniform, Positive, Negative
        labels = np.array([
            [0, 1, 2, 3],
            [3, 2, 1, 0],
        ])
        unary = -np.log(np.array([
            [U, P, N, N, N, N, P, U],
            [U, N, P, N, N, P, N, U],
            [U, N, N, P, P, N, N, U],
        ]))

        np.testing.assert_almost_equal(utils.compute_unary(labels, M, GT_PROB=P), unary)


if __name__ == "__main__":
    unittest.main()
