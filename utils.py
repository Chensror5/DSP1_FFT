import numpy as np

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


