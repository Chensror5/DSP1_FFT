
########################################################################################
#                            DSP1 - Programming Assigment
# By:
#   Yoav Allinson         206036949
#   Chen Yaakov Sror      203531645
########################################################################################

import numpy as np
import matplotlib.pyplot as plt
# from matlabcall_test import dist_image_1, dist_image_2, noised_image, imp_resp_image # after using the matlab engine we diabled it to save time - all the images are saved as npz files
from utils import *

d_1 = 0.206036949
d_2 = 0.203531645
d = (d_1 + d_2) % 0.5

dist_image_1 = np.load('dist_image_1_yoav.npz')['arr_0']
dist_image_2 = np.load('dist_image_2_yoav.npz')['arr_0']
imp_resp_image =np.load('imp_resp_image_yoav.npz')['arr_0']
noised_image =np.load('noised_image_yoav.npz')['arr_0']

print("If value is True - Detimation in time", d % 0.1 < 0.05) # True

print("\n")
#dezimation in time

########################################################################################
#           Introduction Part - Implementation of FFT and IFFT Algorithm 
########################################################################################

def FFT(x, flag = True):
    if flag:
    # Check if the input serie is not a power of 2, and pad it to a power of 2 accordingly
        if not ispowerof2(len(x)):
            zeros_to_pad = getnextpow2(len(x)) - len(x)
            x = np.pad(x, (0, zeros_to_pad), "constant")
            flag = False
            
    N = len(x)
    
    if N == 1:
        return x
    else:

        X_even = FFT(x[::2], flag)
        X_odd = FFT(x[1::2], flag)
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

def linear_convolution_2D(x,h):
    N1 ,M1 = x.shape
    N2,M2 = h.shape
    # total_length_row = max(x.shape[0], h.shape[0])
    # total_length_col = max(x.shape[1], h.shape[1])
    x = np.pad(x, ((0, N1+N2 -1 - x.shape[0]), (0, M1+M2-1 - x.shape[1])), "constant")
    h = np.pad(h, ((0, N1+N2-1 - h.shape[0]), (0, M1+M2-1 - h.shape[1])), "constant")
    X = FFT_2D(x)
    H = FFT_2D(h)
    Y = IFFT_2D(X * H)
    return np.resize(Y,x.shape)

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

mat_x = FFT_2D(x)
magnitude_spectrum = np.abs(mat_x)
plt.imshow(magnitude_spectrum, cmap='turbo')
plt.colorbar()
plt.title('2D DFT Magnitude Spectrum')
plt.savefig('pictures/q1_D_2d_dft.png')
plt.close()

fig, axs = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.2, hspace=0.4)
axs[0, 0].imshow(dist_image_1)
axs[0, 0].set_title('Dist imgge 1', verticalalignment='top')
axs[0, 1].imshow(dist_image_2)
axs[0, 1].set_title('Dist imgge 2', verticalalignment='top')
axs[1, 0].imshow(noised_image)
axs[1, 0].set_title('Noised image', verticalalignment='top')
axs[1, 1].imshow(imp_resp_image)
axs[1, 1].set_title('imp resp image', verticalalignment='top')
plt.savefig('pictures/q1_D_all_images.png')
plt.close()

#   Part 1.e
h_0 = imp_resp_image[:, 0]
samples_vector = np.array([0, 2*np.pi/6, 2* 2*np.pi/6, 4 * 2*np.pi/6])
samples_vector_transposed = np.transpose(samples_vector)
calc_ans = FFT(h_0) * samples_vector_transposed
calc_ans_abs = abs(calc_ans)


plt.stem(calc_ans_abs, basefmt=' ', use_line_collection=True)
plt.title(r'${H}_{0}(e^{j\omega})$')
plt.ylabel("Amplitude")
plt.xlabel(r'${\omega}$')
plt.savefig('pictures/q1_E_H_0_fft.png')
plt.close()
#   Part 1.f

w = np.zeros(32)
w[0] = 1
w[29] = 1

conv_ans = cyclic_convolution(h_0, w)

plt.stem(conv_ans, basefmt=' ', use_line_collection=True)
plt.title(r'${h}_{0} \circledast w[n]$')
plt.ylabel("Amplitude")
plt.xlabel(r'${\omega}$')
plt.savefig('pictures/q1_F_wn_conv_h0n.png')
plt.close()

x1_restored = cyclic_convolution_2D(dist_image_1, imp_resp_image)
x1_restored_abs = np.abs(x1_restored)
plt.imshow(x1_restored_abs,cmap='gray')
plt.title(r'${x_1}[n] \circledast h[n]$')
plt.ylabel("Amplitude")
plt.xlabel(r'${\omega}$')
plt.savefig('pictures/q1_G_x1.png')
plt.close()

fig, axs1 = plt.subplots(1, 2)
plt.subplots_adjust(wspace=0.2, hspace=0.4)
x2_restored = cyclic_convolution_2D(dist_image_2, imp_resp_image)
x2_restored_abs = np.real(x2_restored)
x2_resized = np.resize(x2_restored_abs,(70,170))
x2_restored_abs = x2_restored_abs[:70, :170]
axs1[0].imshow(x2_restored_abs, cmap = 'gray')
axs1[0].set_title(r'${x_2}[n] \circledast h[n]$ With our FFT')


x2_restored_fft2 = np.fft.ifft2(np.fft.fft2(dist_image_2) * np.fft.fft2(np.pad(imp_resp_image, ((0,70-3), (0, 170-5)), 'constant')))
x2_restored_fft2_abs = np.abs(x2_restored_fft2)
x2_restored_fft2_abs = x2_restored_fft2_abs[:70, :170]
axs1[1].imshow(x2_restored_fft2_abs, cmap = 'gray')
axs1[1].set_title(r'${x_2}[n] \circledast h[n]$ With np FFT')

plt.savefig('pictures/q1_G_x2.png')
plt.close()


#### bonus
x2_bonus = linear_convolution_2D(dist_image_2, imp_resp_image)
x2_bonus_abs = np.real(x2_bonus)
x2_bonus_resized = np.resize(x2_bonus_abs,(70,170))
plt.imshow(x2_restored_abs, cmap = 'gray')
plt.suptitle(r'${x_2}[n] \ast h[n]$ With our FFT')
plt.close()

########################################################################################
#           Part 2 - Speech analysis
########################################################################################
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
def create_w_linspace(N):
    return np.concatenate((np.linspace(0,2*np.pi,N)[:int(N/2)+1],np.linspace(-2*np.pi,0,N)[int(N/2)+1:]))

def anti_aliasing(x,dec_factor):
    X = FFT(x)
    a_filter = np.ones_like(x)
    w_m = create_w_linspace(len(x))
    a_filter[np.where(abs(w_m)>np.pi/dec_factor)]=0
    x_filtered =IFFT(X*a_filter).real
    return x_filtered


wav,sr = sf.read('wavs/out.wav')
N =2**16

x_n = np.array(wav[:N])
Px = (1/N) * np.sum(x_n**2)

w1=1.6+0.1*d_1
w2=1.6+0.1*d
w3=3
n = np.arange(N)
z_n = 50*np.sqrt(Px)*(np.cos(w1*n)+np.cos(w2*n)+np.cos(w3*n))

y_n = x_n + z_n
sf.write('wavs/x_n.wav',x_n,16000)
sf.write('wavs/y_n.wav',y_n,16000)
fig = plt.figure()
fig.suptitle(r'$y[n] = x[n] + z[n]$')
plt.plot(y_n)
plt.savefig('pictures/q2_y_n.png')
plt.close()

Y_k = FFT(y_n)
w_m = create_w_linspace(128)
fig = plt.figure()
fig.suptitle(r'$Y(e^{j\omega})$ k= [0,128]')
plt.stem(w_m,abs(Y_k)[::512])
plt.savefig('pictures/q2_Y_ejw.png')
plt.close()

Z_k = FFT(z_n)
w_m = create_w_linspace(len(Z_k))
fig = plt.figure()
fig.suptitle(r'$Z(e^{j\omega})$ k = [0,128]')
plt.stem(w_m,abs(Z_k))
plt.savefig('pictures/q2_Z_ejw.png')
plt.close()



y_n_2 = anti_aliasing(y_n,dec_factor =2)[::2]
sf.write('wavs/y_2.wav',y_n_2,8000)
z_n_2 = anti_aliasing(z_n,2)[::2]
Y_2_k = FFT(y_n_2)
Z_2_k =FFT(z_n_2)
fig = plt.figure()
fig.suptitle(r'$y_{2}[n]$')
plt.plot(y_n_2)
plt.savefig('pictures/q2_F_y2.png')
plt.close()

w_m = create_w_linspace(128)
fig = plt.figure()
fig.suptitle(r'$Y_{2}(e^{j\omega})$ with fft of 256 samples k = [0,128]')
plt.stem(w_m,abs(Y_2_k[::256]))
plt.savefig('pictures/q2_F_Y_2_ejw.png')
plt.close()

fig = plt.figure()
fig.suptitle(r'$Z_{2}(e^{j\omega})$ k = [0,128]')
plt.stem(w_m,abs(Z_2_k[::256]))
plt.savefig('pictures/q2_F_Z_2_ejw.png')
plt.close()

