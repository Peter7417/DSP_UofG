import numpy as np
import matplotlib.pyplot as plt
import firfilter

# Q3
"""Load data into python"""
data = np.loadtxt('ECG_msc_matric_5.dat')

"""Define constants"""
fs = 250  # sampling frequency of data
taps = fs * 2  # number of taps assigned for system
f0 = 50  # noise signal frequency
lR = 0.00089  # learning rate of the LMS filter
t_max = len(data) / fs  # sample time for data

w = np.empty(len(data))  # create an empty array to store LMS filter results
t = np.linspace(0, t_max, len(w))  # create an array to model the x-axis with time values

"""Call the dofilteradaptive function from the class to compute the FIR dataset"""
f = firfilter.firFilter(np.zeros(taps))
for i in range(len(data)):
    sinusoid = (np.sin(2 * np.pi * i * (f0 / fs)))
    w[i] = f.dofilterAdaptive(data[i], sinusoid, lR)


"""Plot the LMS filter"""
plt.plot(t, w)
plt.title('ECG 50Hz LMS Filter')
plt.xlabel('Time (sec)')
plt.ylabel('ECG (volts)')

plt.show()
