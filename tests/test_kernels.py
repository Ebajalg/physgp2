import pytest
import numpy as np
from physgp2.kernels import Kernel, MatrixKernel


def test_kernel():
    x1, x2 = np.array([1,2,3]), np.array([1,2,3])
    kernel = Kernel(lambda x1, x2: np.dot(x1, x2))
    assert np.allclose(kernel(x1, x2), np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]]))


def test_matrix_kernel_init_1():
    with pytest.raises(TypeError) as exp:
        k1 = Kernel(lambda x1, x2: np.dot(x1, x2))
        MatrixKernel(k1)
    
    assert str(exp.value) == "Input must be a numpy 2D array of kernel functions."
    

def test_matrix_kernel_init_2():
    with pytest.raises(TypeError) as exp:
        k1 = Kernel(lambda x1, x2: np.dot(x1, x2))
        k2 = Kernel(lambda x1, x2: np.sum(x1 * x2))
        kernels_matrix = np.array([[k1, k2], [k2, lambda x1, x2: np.linalg.norm(x1 - x2)]])
        MatrixKernel(kernels_matrix)

    assert str(exp.value) == "All elements of the matrix must be instances of the Kernel class."