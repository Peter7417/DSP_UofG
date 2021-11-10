import numpy as np
import matplotlib.pyplot as plt
import firfilter

"""Reshuffle Function"""


def reshuffle(filter_coeff):
    h = np.zeros(taps)
    h[0:int(taps / 2)] = filter_coeff[int(taps / 2):taps]
    h[int(taps / 2):taps] = filter_coeff[0:int(taps / 2)]
    return h * np.hanning(taps)


"""50 HZ removal"""


def bandstopDesign(samplerate, w1, w2):
    # frequency resolution =0.5
    M = samplerate * 2
    cutoff_1 = w1
    cutoff_2 = w2
    X = np.ones(M)
    X[cutoff_1:cutoff_2 + 1] = 0
    X[M - cutoff_2:M - cutoff_1 + 1] = 0
    x = np.real(np.fft.ifft(X))

    return x


"""DC noise removal"""


def highpassDesign(samplerate, w3):
    # frequency resolution =0.5
    M = samplerate * 2
    cutoff = w3
    X = np.ones(M)
    X[0:cutoff + 1] = 0
    X[M - cutoff: M + 1] = 0
    x = np.real(np.fft.ifft(X))

    return x


# Q1 and Q2
"""Load data into python"""
data = np.loadtxt('ECG_msc_matric_5.dat')

"""Define constants"""
fs = 250  # sample frequency
t_max = len(data) / fs  # sample time of data
t = np.linspace(0, t_max, len(data))  # create an array to model the x-axis
taps = (fs * 2)  # defining taps

"""Bandstop"""
f1 = int((45 / fs) * taps)  # before 50Hz
f2 = int((55 / fs) * taps)  # after 50Hz

"""Highpass"""
f3 = int((0.5 / fs) * taps)  # ideal for cutting off DC noise

"""Call the function for Bandstop and Highpass"""
impulse_BS = bandstopDesign(fs, f1, f2)
impulse_HP = highpassDesign(fs, f3)

"""Reshuffle the coefficients for highpass by calling reshuffle function"""
h_newHP = reshuffle(impulse_HP)

"""Reshuffle the coefficients for bandstop by calling reshuffle function"""
h_newBS = reshuffle(impulse_BS)

"""Call the class method dofilter, by passing in only a scalar value at a time which outputs a scalar value"""
# obtain FIR_HP output when we couple the original ECG data with the highpass over a ring buffer
fir_HP = np.empty(len(data))
fi = firfilter.firFilter(h_newHP)
for i in range(len(fir_HP)):
    fir_HP[i] = fi.dofilter(data[i])

# obtain FIR output when we couple the previously found FIR_HP data with the bandstop over a ring buffer
fir = np.empty(len(data))
po = firfilter.firFilter(h_newBS)
for i in range(len(fir)):
    fir[i] = po.dofilter(fir_HP[i])

"""Plot both the original ECG data set and new filtered data set """
plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(t, data)
plt.title('ECG')
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

t1 = np.linspace(0, t_max, len(fir))
plt.subplot(1, 2, 2)
plt.plot(t1, fir)
plt.xlim(0, t_max)
plt.title('ECG 50Hz and Dc Noise Removed')
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

plt.show()
