
########################################################################################
#                            DSP1 - Programming Assigment
# By:
#   Yoav Allinson         206036949
#   Chen Yaakov Sror      203531645
########################################################################################

import numpy as np
import matplotlib.pyplot as plt
from test import *
from matlabcall_test import dist_image_1, dist_image_2, noised_image, imp_resp_image

d_1 = 0.206036949
d_2 = 0.203531645

d = (d_1 + d_2)%0.5

print("If value is True - Detimation in time", d%0.1 < 0.05) # True
print("\n")
#dezimation in time

########################################################################################
#                               Helper Functions  
########################################################################################

def ispowerof2(num):
    while (num %2 == 0):
        num = num / 2
    if num == 1:
        return True
    else:
        return False

def getnextpow2(num):
    power = 0
    while(2 ** power < num):
        power += 1
    return 2 ** power

def DFT(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

########################################################################################
#           Introduction Part - Implementation of FFT and IFFT Algorithm 
########################################################################################
def FFT(x):
    """
    A recursive implementation of 
    the 1D Cooley-Tukey FFT, the 
    input should have a length of 
    power of 2. 
    """
    if not ispowerof2(len(x)):
        zeros_to_pad = getnextpow2(len(x)) - len(x)
        x = np.pad(x, (0, zeros_to_pad), "constant")
    
    N = len(x)
    
    if N == 1:
        return x
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j*np.pi*np.arange(N)/ N)
        
        X = np.concatenate([X_even+factor[:int(N/2)]*X_odd, X_even+factor[int(N/2):]*X_odd])
        return X
    
def IFFT(x):
    N = len(x)
    return (1/N)*FFT(x.conj).conj()

########################################################################################
#           Part 1 - Image analysis and processing using two-dimensional DTFT.
########################################################################################

B1 = 7
B2 = 7
M = 64
N = 32
x = np.zeros((N,M))

for i in range(N):
    for j in range(M):
        if (0 <= i and i < B1) and (0 <= j and j < B2):
            x[i][j] = 1
        else:
            x[i][j] = 0

def FFT_2D(image):
    row, col = image.shape
    mat = np.zeros_like(image, dtype=np.complex128)
    
    for i in range(col):
        mat[:, i] = FFT(image[:, i])
    for i in range(row):
        mat[i, :] = FFT(mat[i, :])
    return mat

return_mat = FFT_2D(x)
magnitude_spectrum = np.abs(return_mat)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.colorbar()
plt.title('2D DFT Magnitude Spectrum')
plt.show()

return_mat = FFT_2D(dist_image_1)
magnitude_spectrum = np.abs(return_mat)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.colorbar()
plt.title('2D DFT Magnitude Spectrum')
plt.show()

return_mat = FFT_2D(dist_image_2)
magnitude_spectrum = np.abs(return_mat)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.colorbar()
plt.title('2D DFT Magnitude Spectrum')
plt.show()
# 
# return_mat = FFT_2D(imp_resp_image)
# magnitude_spectrum = np.abs(return_mat)
# plt.imshow(magnitude_spectrum, cmap='gray')
# plt.colorbar()
# plt.title('2D DFT Magnitude Spectrum')
# plt.show()
# 
# 