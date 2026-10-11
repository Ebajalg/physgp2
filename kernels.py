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

    def invK(self, X1, X2):
        """
        Compute the inverse of the kernel matrix between two sets of input vectors.
        As K is a covariance matrix, it must be square and symmetric. 
        This method uses the FFT to compute the inverse efficiently for circulant matrices.

        Parameters:
        X1: A 2D array-like structure where each row is a sample.
        X2: A 2D array-like structure where each row is a sample.

        Returns:
        A 2D array representing the inverse of the kernel matrix.
        """
        K = self(X1, X2)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("A covariance matrix must be square to be inverted.")
        if not np.allclose(K, K.T):
            raise ValueError("The covariance matrix must be symmetric.")

        # The FFT diagonalizes circulant matrices, but not arbitrary symmetric
        # covariance matrices. Verify this structure before using the FFT.
        n = K.shape[0]
        circulant = np.vstack([np.roll(K[0], i) for i in range(n)])
        if not np.allclose(K, circulant):
            raise ValueError("FFT inversion requires a circulant covariance matrix.")

        eigenvalues = np.fft.fft(K[0])
        if np.any(np.isclose(eigenvalues, 0)):
            raise np.linalg.LinAlgError("Covariance matrix is singular.")
        inverse_first_row = np.fft.ifft(1.0 / eigenvalues).real

        return K, np.vstack([np.roll(inverse_first_row, i) for i in range(n)])




class MatrixKernel:
    def __init__(self, kernels : np.matrix):
        # Check that the input is a valid matrix of kernel functions
        if not isinstance(kernels, np.matrix):
            raise TypeError("Input must be a numpy matrix of kernel functions.")

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
        

    def invK(self, X1_vec, X2_vec):
        """
        Compute the inverse of the block kernel matrix for a matrix of kernel functions.

        Parameters:
        X1_vec: A list of 2D array-like structures where each element is a vector 
        of samples corresponding to the first set of inputs for each kernel function.

        X2_vec: A list of 2D array-like structures where each element is a vector 
        of samples corresponding to the second set of inputs for each kernel function.


        Returns:
        A 2D array representing the inverse of the block kernel matrix.
        """
        K = self(X1_vec, X2_vec)
        return K, np.linalg.inv(K)