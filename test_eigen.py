import numpy as np
import eigen as e

V = np.random.randn(3).astype(np.float32)
M = np.random.randn(3,3).astype(np.float32)

foo = e.vectorXf(V)
assert np.all(np.array(foo) == V)

foo = e.matrixXf(M)
assert np.all(np.array(foo) == M)
