import numpy as np
import matplotlib.pyplot as plt
from matlabcall_test import dist_image_1, dist_image_2, noised_image, imp_resp_image


def dft_2d(matrix):
    M, N = matrix.shape
    dft_matrix = np.zeros_like(matrix, dtype=np.complex128)

    for u in range(M):
        for v in range(N):
            for x in range(M):
                for y in range(N):
                    angle = 2 * np.pi * ((u * x / M) + (v * y / N))
                    dft_matrix[u, v] += matrix[x, y] * np.exp(-1j * angle)

    return dft_matrix

# Example usage
dist_image_1

return_mat = dft_2d(dist_image_1)
magnitude_spectrum = np.abs(return_mat)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.colorbar()
plt.title('2D DFT Magnitude Spectrum')
plt.show()
