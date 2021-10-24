import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


def bandstopDesign(f, w1, w2):
    taps = f * 2
    cutoff_1 = int(w1 * taps)
    cutoff_2 = int(w2 * taps)
    X = np.ones(taps)
    X[cutoff_1:cutoff_2+1] = 0
    X[taps-cutoff_2:taps-cutoff_1+1] = 0
    x = np.real(np.fft.ifft(X))
    h = np.zeros(taps)
    h[0:int(taps/2)] = x[int(taps/2):taps]
    h[int(taps/2):taps] = x[0:int(taps/2)]
    h_new = h * np.blackman(taps)

    # """Plot original bandpass"""
    # plt.figure(3)
    # plt.subplot(2, 2, 1)
    # plt.plot(np.linspace(0, f, len(X)), X)
    # plt.title('Original')
    #
    # """Invert obtained h_new to verify consistency"""
    # hf = np.abs(np.fft.fft(h_new))
    # plt.subplot(2, 2, 2)
    # plt.plot(np.linspace(0, f, len(hf)), 20*np.log10(hf))
    # plt.title('FFT recreation')
    #
    # """Plot the h and h_new to show windowing effect"""
    # plt.subplot(2, 2, 3)
    # plt.plot(h)
    # plt.title('Original impulse response')
    # plt.subplot(2, 2, 4)
    # plt.plot(h_new)
    # plt.title('Windowed impulse response')
    # plt.suptitle('bandpass')
    return h_new


"""DC noise removal"""


def highpassDesign(f, w2):
    taps = f * 2
    cutoff = int(w2 * taps)
    X = np.ones(taps)
    X[0:cutoff+1] = 0
    x = np.real(np.fft.ifft(X))
    h = np.zeros(taps)
    h[0:int(taps / 2)] = x[int(taps / 2):taps]
    h[int(taps / 2):taps] = x[0:int(taps / 2)]
    h_new = h * np.blackman(taps)

    # """Plot original highpass"""
    # plt.figure(2)
    # plt.subplot(2, 2, 1)
    # plt.plot(np.linspace(0, f, len(X)), X)
    # plt.title('Original')
    #
    # """Invert obtained h_new to verify consistency"""
    # hf = np.abs(np.fft.fft(h_new))
    # plt.subplot(2, 2, 2)
    # plt.plot(np.linspace(0, f, len(hf)), 20 * np.log10(hf))
    # plt.title('FFT recreation')
    #
    # """Plot the h and h_new to show windowing effect"""
    # plt.subplot(2, 2, 3)
    # plt.plot(h)
    # plt.title('Original impulse response')
    # plt.subplot(2, 2, 4)
    # plt.plot(h_new)
    # plt.title('Windowed impulse response')
    # plt.suptitle('highpass')
    return h_new


"""Plot the ECG"""
data = np.loadtxt('ECG_msc_matric_5.dat')
t_max = len(data) * 20
t = np.linspace(0, t_max, len(data))
plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(t, data)
plt.title('ECG')
plt.xlabel('time(sec)')
plt.ylabel('ECG raw')

fs = 250
f1 = 45/fs
f2 = 55/fs
f3 = 0.1/fs


impulse_BS = bandstopDesign(fs, f1, f2)
impulse_HP = highpassDesign(fs, f3)
res = np.convolve(impulse_HP, impulse_BS)
conv = np.convolve(res, data, mode='full')
y = signal.lfilter(res,1,data)
t1 = np.linspace(0, len(conv)*20, len(conv))
plt.figure(1)
plt.subplot(1, 2, 2)
plt.plot(t1, conv)
# plt.xlim(0, t_max)
plt.title('ECG 50Hz Removed using np.convolve')
plt.xlabel('time(sec)')
plt.ylabel('ECG raw')

plt.figure(4)
t2 = np.linspace(0, len(y)*20, len(y))
plt.title('using lfilter')
plt.plot(t2,y)

plt.show()
