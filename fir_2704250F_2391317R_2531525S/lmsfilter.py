import numpy as np
import matplotlib.pyplot as plt
import firfilter

"""Reshuffle Function"""


def reshuffle(filter_coeff):
    N_taps = len(filter_coeff)
    h = np.zeros(N_taps)  # create an array to hold the impulse response values
    h[0:int(N_taps / 2)] = filter_coeff[int(N_taps / 2):N_taps]  # perform a reshuffling action
    h[int(N_taps / 2):N_taps] = filter_coeff[0:int(N_taps / 2)]  # perform a reshuffling action
    return h   # return the impulse response


"""50 HZ removal"""


def bandstopDesign(samplerate, w1, w2, margin):
    taps = int(np.abs((samplerate / (((w1 + w2) / 2) - w1))))  # calculate the ntaps
    M = taps * margin  # account for the transition width using our predefined margin
    X = np.ones(M)  # create an array of ones to model an ideal bandstop
    cutoff_1 = int((w1 / samplerate) * M)  # array index calculation for cutoff frequency 1
    cutoff_2 = int((w2 / samplerate) * M)  # array index calculation for cutoff frequency 2
    X[cutoff_1:cutoff_2 + 1] = 0  # mirror 1 (set all values to 0)
    X[M - cutoff_2:M - cutoff_1 + 1] = 0  # mirror 2 (set all values to 0)
    x = np.real(np.fft.ifft(X))  # perform IDFT to obtain an a-causal system

    return x


"""Load data into python"""
data = np.loadtxt('ecg.dat')

"""Define constants"""
fs = 250  # sample frequency
t_max = len(data) / fs  # sample time of data
t_data = np.linspace(0, t_max, len(data))  # create an array to model the x-axis with time values
f_sine = 50  # noise signal frequency
lR = 0.01  # learning rate of the LMS filter
transition_width_compensation = 2  # to account for the transition width in a practical scenario by a factor

"""Bandstop"""
f1 = 45  # cutoff frequency before 50Hz
f2 = 55  # cutoff frequency after 50Hz

"""Call the function for Bandstop and Highpass"""
impulse_BS = bandstopDesign(fs, f1, f2, transition_width_compensation)

"""Reshuffle the time_reversed_coeff for bandstop by calling reshuffle function"""
h_newBS = reshuffle(impulse_BS) * np.blackman(len(impulse_BS))  # apply the appropriate window function

"""Call the class method dofilter, where we only perform 50Hz removal to compare with our LMS filter"""
fir_BS = np.empty(len(data))
fi = firfilter.firFilter(h_newBS)
for i in range(len(fir_BS)):
    fir_BS[i] = fi.dofilter(data[i])

# Q3

lms = np.empty(len(data))  # create an empty array to store LMS filter results
time = np.linspace(0, t_max, len(lms))  # create an array to model the x-axis with time values

"""Call the dofilterAdaptive function from the class to compute the FIR dataset"""
f = firfilter.firFilter(np.zeros(len(impulse_BS)))
for i in range(len(data)):
    sinusoid = (np.sin(2 * np.pi * i * (f_sine / fs)))
    lms[i] = f.dofilterAdaptive(data[i], sinusoid, lR)

"""Plot the LMS filter"""
plt.figure(1)
plt.plot(time, lms, label='LMS filter')
plt.plot(time, fir_BS, label='Bandstop 50Hz')
plt.title('Bandstop 50Hz Filter and LMS Filter @ learning rate = ' + str(lR))
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')
plt.legend(loc='upper right')

plt.figure(2)
plt.plot(time, lms, label='LMS filter')
plt.title('LMS Filter @ learning rate = ' + str(lR))
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')
plt.legend(loc='upper right')

plt.show()
