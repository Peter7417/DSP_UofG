import numpy as np
import matplotlib.pyplot as plt
import firfilter

"""Reshuffle Function"""


def reshuffle(filter_coeff):
    N_taps = len(filter_coeff)
    h = np.zeros(N_taps)  # create an array to hold the impulse response values
    h[0:int(N_taps / 2)] = filter_coeff[int(N_taps / 2):N_taps]  # perform a reshuffling action
    h[int(N_taps / 2):N_taps] = filter_coeff[0:int(N_taps / 2)]  # perform a reshuffling action
    return h * np.hanning(N_taps)  # return the impulse response with a window function applied to it


"""50 HZ removal"""


def bandstopDesign(samplerate, w1, w2):
    # frequency resolution =0.5
    M = samplerate * 2  # calculate the ntaps
    X = np.ones(M)  # create an array of ones to model an ideal bandstop
    cutoff_1 = int((w1 / samplerate) * M)  # array index calculation for cutoff frequency 1
    cutoff_2 = int((w2 / samplerate) * M)  # array index calculation for cutoff frequency 2
    X[cutoff_1:cutoff_2 + 1] = 0  # mirror 1 (set all values to 0)
    X[M - cutoff_2:M - cutoff_1 + 1] = 0  # mirror 2 (set all values to 0)
    x = np.real(np.fft.ifft(X))  # perform IDFT to obtain an a-causal system

    return x


"""DC noise removal"""


def highpassDesign(samplerate, w3):
    # frequency resolution =0.5
    M = samplerate * 2  # calculate the ntaps
    X = np.ones(M)  # create an array of ones to model an ideal highpass
    cutoff_3 = int((w3 / samplerate) * M)  # array index calculation for cutoff frequency 3
    X[0:cutoff_3 + 1] = 0  # mirror 1 (set all values to 0)
    X[M - cutoff_3: M + 1] = 0  # mirror 2 (set all values to 0)
    x = np.real(np.fft.ifft(X))  # perform IDFT to obtain an a-causal system

    return x


# Q1 and Q2
"""Load data into python"""
data = np.loadtxt('ecg.dat')

"""Define constants"""
fs = 250  # sample frequency
t_max = len(data) / fs  # sample time of data
t_data = np.linspace(0, t_max, len(data))  # create an array to model the x-axis with time values
resolution_factor = 2  # define by how much the ntaps are greater than the sampling rate to account for transition width
ntaps = (fs * resolution_factor)  # defining ntaps

"""Bandstop"""
f1 = 45  # cutoff frequency before 50Hz
f2 = 55  # cutoff frequency after 50Hz

"""Highpass"""
f3 = 0.5  # ideal for cutting off DC noise

"""Call the function for Bandstop and Highpass"""
impulse_BS = bandstopDesign(fs, f1, f2)
impulse_HP = highpassDesign(fs, f3)

"""Reshuffle the time_reversed_coeff for highpass by calling reshuffle function"""
h_newHP = reshuffle(impulse_HP)

"""Reshuffle the time_reversed_coeff for bandstop by calling reshuffle function"""
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
plt.plot(t_data, data)
plt.title('ECG')
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

plt.subplot(1, 2, 2)
plt.plot(t_data, fir)
plt.xlim(0, t_max)
plt.title('ECG 50Hz and Dc Noise Removed')
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

plt.show()
