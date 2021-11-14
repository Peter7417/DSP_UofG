import numpy as np
import matplotlib.pyplot as plt
import firfilter

"""Reshuffle Function"""


def reshuffle(filter_coeff):
    h = np.zeros(taps)  # create an array to hold the impulse response values
    h[0:int(taps / 2)] = filter_coeff[int(taps / 2):taps]  # perform a reshuffling action
    h[int(taps / 2):taps] = filter_coeff[0:int(taps / 2)]  # perform a reshuffling action
    return h * np.hanning(taps)  # return the impulse response with a window function applied to it


"""50 HZ removal"""


def bandstopDesign(samplerate, w1, w2, itr):
    # frequency resolution =0.5
    M = samplerate * itr  # calculate the taps
    X = np.ones(M)  # create an array of ones to model an ideal bandstop
    X[w1:w2 + 1] = 0  # mirror 1 (set all values to 0)
    X[M - w2:M - w1 + 1] = 0  # mirror 2 (set all values to 0)
    x = np.real(np.fft.ifft(X))  # perform IDFT to obtain an a-causal system

    return x


"""Load data into python"""
data = np.loadtxt('ECG_msc_matric_5.dat')

"""Define constants"""
fs = 250  # sample frequency
t_max = len(data) / fs  # sample time of data
t_data = np.linspace(0, t_max, len(data))  # create an array to model the x-axis with time values
practical = 2  # define by how much the taps are greater than the sampling rate to account for transition width
taps = (fs * practical)  # defining taps
f_sine = 50  # noise signal frequency
lR = 0.001  # learning rate of the LMS filter

"""Bandstop"""
f1 = int((45 / fs) * taps)  # cutoff frequency before 50Hz
f2 = int((55 / fs) * taps)  # cutoff frequency after 50Hz

"""Call the function for Bandstop and Highpass"""
impulse_BS = bandstopDesign(fs, f1, f2, practical)

"""Reshuffle the time_reversed_coeff for bandstop by calling reshuffle function"""
h_newBS = reshuffle(impulse_BS)

"""Call the class method dofilter, where we only perform 50Hz removal to compare with our LMS filter"""
fir_BS = np.empty(len(data))
fi = firfilter.firFilter(h_newBS)
for i in range(len(fir_BS)):
    fir_BS[i] = fi.dofilter(data[i])

# Q3

lms = np.empty(len(data))  # create an empty array to store LMS filter results
time = np.linspace(0, t_max, len(lms))  # create an array to model the x-axis with time values

"""Call the dofilterAdaptive function from the class to compute the FIR dataset"""
f = firfilter.firFilter(np.zeros(taps))
for i in range(len(data)):
    sinusoid = (np.sin(2 * np.pi * i * (f_sine / fs)))
    lms[i] = f.dofilterAdaptive(data[i], sinusoid, lR)

"""Plot the LMS filter"""
plt.figure(1)
plt.plot(time, lms, label='LMS filter')
plt.plot(time, fir_BS, label='Bandstop 50Hz')
plt.title('Bandstop 50Hz Filter and LMS Filter @ learning rate = ' + str(lR))
plt.xlabel('Time (sec)')
plt.ylabel('ECG (volts)')
plt.legend(loc='upper right')

plt.figure(2)
plt.plot(time, lms, label='LMS filter')
plt.title('LMS Filter @ learning rate = ' + str(lR))
plt.xlabel('Time (sec)')
plt.ylabel('ECG (volts)')
plt.legend(loc='upper right')

plt.show()
