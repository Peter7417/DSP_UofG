import numpy as np
import matplotlib.pyplot as plt
import filter as fil


def convolve(x,y):

    ls = []
    l1 = len(x)
    l2 = len(y)
    N = l1 + l2 - 1
    k = np.zeros(N)

    for n in range(N):
        for i in range(l1):
            if 0 <= (n - i + 1) < l2:
                k[n] = k[n] + x[i] * y[n - i + 1]

    for i in k:
        ls.append(i)

    ls.insert(0, ls.pop())
    ls[0] = x[0]
    return np.array(ls)


def bandstopDesign(f, w1, w2):
    taps = f * 2    # frequency resolution =0.5
    cutoff_1 = int(w1 * taps)
    cutoff_2 = int(w2 * taps)
    X = np.ones(taps)
    X[cutoff_1:cutoff_2 + 1] = 0
    X[taps - cutoff_2:taps - cutoff_1 + 1] = 0
    x = np.real(np.fft.ifft(X))

    return x


"""DC noise removal"""


def highpassDesign(f, w2):
    taps = f * 2    # frequency resolution =0.5
    cutoff = int(w2 * taps)
    X = np.ones(taps)
    X[0:cutoff + 1] = 0
    x = np.real(np.fft.ifft(X))

    return x


"""Plot the ECG"""
data = np.loadtxt('ECG_msc_matric_5.dat')
t_max = len(data) * 20
t = np.linspace(0, t_max, len(data))
M = 500
plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(t, data)
plt.title('ECG')
plt.xlabel('time(sec)')
plt.ylabel('ECG raw')

fs = 250  # sample frequency

"""Bandstop"""
f1 = 45 / fs  # before 50Hz
f2 = 55 / fs  # after 50Hz

"""Highpass"""
f3 = 0.33 / fs  # ideal for cutting off DC noise

"""Call the function for Bandstop and Highpass"""
impulse_BS = bandstopDesign(fs, f1, f2)
impulse_HP = highpassDesign(fs, f3)

"""Convolve used here since there are a large number of calculations used (James and Nick)"""
# inverse = np.convolve(impulse_HP, impulse_BS, mode='full')
inverse = convolve(impulse_HP, impulse_BS)

"""Call the class get the output with 50Hz and DC noise removal"""
fir = fil.firFilter(fs, data).dofilter(inverse)


t1 = np.linspace(0, len(fir) * 20, len(fir))
plt.figure(1)
plt.subplot(1, 2, 2)
plt.plot(t1, fir)
plt.title('ECG 50Hz and Dc Noise Removed')
plt.xlabel('time(sec)')
plt.ylabel('ECG raw')
# plt.figure(2)
# plt.plot(h_new)


plt.show()
