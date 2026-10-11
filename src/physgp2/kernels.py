import numpy as np

class Kernel:
    def __init__(self, u_func):
        self.u_func = u_func


    def get_info(self):
        return f"Kernel Function: {self.u_func}"


    def __call__(self, X1, X2):
        """
        Compute the kernel matrix between two sets of input vectors.

        Parameters:
        X1: A 2D array-like structure where each row is a sample.
        X2: A 2D array-like structure where each row is a sample.

        Returns:
        A 2D array representing the kernel matrix.
        """
        n_samples_1 = X1.shape[0]
        n_samples_2 = X2.shape[0]
        K = np.zeros((n_samples_1, n_samples_2))

        for i in range(n_samples_1):
            for j in range(n_samples_2):
                K[i, j] = self.u_func(X1[i], X2[j])

        return K




class MatrixKernel:
    def __init__(self, kernels : np.ndarray):
        # Check that the input is a valid matrix of kernel functions
        if not isinstance(kernels, np.ndarray) or kernels.ndim != 2:
            raise TypeError("Input must be a numpy 2D array of kernel functions.")

        # Check matrix is full of Kernel instances
        for i in range(kernels.shape[0]):
            for j in range(kernels.shape[1]):
                if not isinstance(kernels[i, j], Kernel):
                    raise TypeError("All elements of the matrix must be instances of the Kernel class.")
        
        self.kernels = kernels


    def get_info(self):
        return f"Matrix Kernel Function: {self.kernels}"


    def __call__(self, X1_vec, X2_vec):
        """
        Compute the block kernel matrix for a matrix of kernel functions.

        Parameters:
        X1_vec: A list of 2D array-like structures where each element is a vector
        of samples corresponding to the first set of inputs for each kernel function.

        X2_vec: A list of 2D array-like structures where each element is a vector
        of samples corresponding to the second set of inputs for each kernel function.


        Returns:
        A 2D array representing the block kernel matrix.
        """
        K_block = []
        for i,X1 in enumerate(X1_vec):
            K_row = []
            for j, X2 in enumerate(X2_vec):
                K = self.kernels[i][j](X1, X2)
                K_row.append(K)
            K_block.append(K_row)

        return np.block(K_block)
        
