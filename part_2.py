d_1 = 0.206036949
d_2 = 0.203531645
d = (d_1 + d_2)%0.5


########################################################################################
#           Part 2 - Speech analysis
########################################################################################
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

wav,sr = sf.read('out.wav')
N =2**16


x_n = np.array(wav[:N])
Px = 1/N * np.sum(x_n**2)

w1=1.6+0.1*d_1
w2=1.6+0.1*d
w3=3
n = np.arange(N)
z_n = 50*np.sqrt(Px)*(np.cos(w1*n)+np.cos(w2*n)+np.cos(w3*n))

y_n = x_n + z_n
sf.write('x_n.wav',x_n,16000)
sf.write('y_n.wav',y_n,16000)
# fig = plt.figure()
# fig.suptitle(r'$y[n] = x[n] + z[n]$')
# plt.plot(y_n)
# plt.savefig('pics/q2_y_n.png')

Y_k = np.fft.fft(y_n,256)
w_m = np.linspace(-np.pi,np.pi,256)
# fig = plt.figure()
# fig.suptitle(r'$Y(e^{j\omega})$ with fft of 256 samples k = [0,128]')
# plt.stem(w_m,Y_k)
# plt.savefig('pics/q2_Y_ejw.png')

y_n_2 = y_n[::2]
z_n_2 = z_n[::2]
Y_2_k = np.fft.fft(y_n_2,256)

# fig = plt.figure()
# fig.suptitle(r'$y_{2}[n]$')
# plt.plot(y_n_2)
# plt.show()


fig = plt.figure()
fig.suptitle(r'$Y_{2}(e^{j\omega})$ with fft of 256 samples k = [0,128]')
plt.stem(w_m,Y_2_k)
plt.show()
sf.write('y_2.wav',y_n_2,8000)
