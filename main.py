
########################################################################################
#                            DSP1 - Programming Assigment
# By:
#   Yoav Allinson         206036949
#   Chen Yaakov Sror      203531645
########################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matlabcall_test import dist_image_1, dist_image_2, noised_image, imp_resp_image
from utils import *

d_1 = 0.206036949
d_2 = 0.203531645
d = (d_1 + d_2) % 0.5

print("If value is True - Detimation in time", d%0.1 < 0.05) # True
print("\n")
#dezimation in time

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
    return (1/N)*FFT(x.conj()).conj()


######## move to utils! #######

def cyclic_convolution(x,h):
    total_length = max(len(x), len(h))
    x = np.pad(x, (0, total_length - len(x)), 'constant')
    h = np.pad(h, (0, total_length - len(h)), 'constant')
    X = FFT(x)
    H = FFT(h)
    Y = IFFT(X * H)
    return Y

def cyclic_convolution_2D(x,h):
    total_length_row = max(x.shape[0], h.shape[0])
    total_length_col = max(x.shape[1], h.shape[1])
    x = np.pad(x, ((0, total_length_row - x.shape[0]), (0, total_length_col - x.shape[1])), "constant")
    h = np.pad(h, ((0, total_length_row - h.shape[0]), (0, total_length_col - h.shape[1])), "constant")
    X = FFT_2D(x)
    H = FFT_2D(h)
    Y = IFFT_2D(X * H)
    return Y
######## move to utils! #######




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

#   Part 1.d
def FFT_2D(x):
    N, M = x.shape[0], x.shape[1]
    N, M = getnextpow2(N), getnextpow2(M)
    mat = np.zeros((N, M), dtype=np.complex128)
    for i in range(x.shape[1]):
        mat[:, i] = FFT(x[:, i])
    for i in range(x.shape[0]):
        mat[i, :] = FFT(mat[i, :])
    return mat

def IFFT_2D(x):
    N, M = x.shape[0], x.shape[1]
    return ( 1 / (N*M) ) * FFT_2D(x.conj()).conj()

# mat_x = FFT_2D(x)
# magnitude_spectrum = np.abs(mat_x)
# plt.imshow(magnitude_spectrum, cmap='turbo')
# plt.colorbar()
# plt.title('2D DFT Magnitude Spectrum')
# plt.show()


# fig, axs = plt.subplots(2, 2)
# plt.subplots_adjust(wspace=0.2, hspace=0.4)
# axs[0, 0].imshow(dist_image_1)
# axs[0, 0].set_title('Dist imgge 1', verticalalignment='top')
# axs[0, 1].imshow(dist_image_2)
# axs[0, 1].set_title('Dist imgge 2', verticalalignment='top')
# axs[1, 0].imshow(noised_image)
# axs[1, 0].set_title('Noised image', verticalalignment='top')
# axs[1, 1].imshow(imp_resp_image)
# axs[1, 1].set_title('imp resp image', verticalalignment='top')
# 
#plt.show()

#   Part 1.e
h_0 = imp_resp_image[:, 0]
samples_vector = np.array([0, 2*np.pi/6, 2* 2*np.pi/6, 4 * 2*np.pi/6])
samples_vector_transposed = np.transpose(samples_vector)
calc_ans = FFT(h_0) * samples_vector_transposed
calc_ans_abs = abs(calc_ans)


# plt.stem(calc_ans_abs, basefmt=' ', use_line_collection=True)
# plt.title(r'${H}_{0}(e^{j\omega})$')
# plt.ylabel("Amplitude")
# plt.xlabel(r'${\omega}$')
# plt.show()
# print("check")
#for ax in axs.flat:
#    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
#for ax in axs.flat:
#    ax.label_outer()


#   Part 1.f

w = np.zeros(32)
w[0] = 1
w[29] = 1

conv_ans = cyclic_convolution(h_0, w)

#plt.stem(conv_ans, basefmt=' ', use_line_collection=True)
#plt.title(r'${h}_{0} \circledast w[n]$')
#plt.ylabel("Amplitude")
#plt.xlabel(r'${\omega}$')
#plt.show()


x1_restored = cyclic_convolution_2D(dist_image_1, imp_resp_image)
x1_restored_abs = np.abs(x1_restored)
plt.imshow(x1_restored_abs)
plt.title(r'${h}_{0} \circledast w[n]$')
plt.ylabel("Amplitude")
plt.xlabel(r'${\omega}$')
plt.show()


fig, axs1 = plt.subplots(1, 2)
plt.subplots_adjust(wspace=0.2, hspace=0.4)

x2_restored = cyclic_convolution_2D(dist_image_2, imp_resp_image)
x2_restored_abs = np.abs(x2_restored)
x2_restored_abs = x2_restored_abs[:70, :170]
axs1[0].imshow(x2_restored_abs, cmap = 'gray')
axs1[0].set_title(r'${h}_{0} our own w[n]$')


print("here")



x2_restored_fft2 = np.fft.ifft2(np.fft.fft2(dist_image_2) * np.fft.fft2(np.pad(imp_resp_image, ((0,70-3), (0, 170-5)), 'constant')))
x2_restored_fft2_abs = np.abs(x2_restored_fft2)
x2_restored_fft2_abs = x2_restored_fft2_abs[:70, :170]
axs1[1].imshow(x2_restored_fft2_abs, cmap = 'gray')
axs1[1].set_title(r'${h}_{0} fft func w[n]$')

print("here")
  


plt.show()
