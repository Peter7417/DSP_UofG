import numpy as np
import matplotlib.pyplot as plt
import firfilter

# Q3
fs = 250
taps = fs * 2
f0 = 50
# dataRaw = np.loadtxt('ECG_msc_matric_4.dat')
# data = dataRaw / max(dataRaw)
data = np.loadtxt('ECG_msc_matric_4.dat')
lR = 0.00089
t_max = len(data) / fs

w = np.empty(len(data))
t = np.linspace(0, t_max, len(w))

f = firfilter.firFilter(np.zeros(taps))
for i in range(len(data)):
    sinusoid = (np.sin(2 * np.pi * i * (f0 / fs)))
    w[i] = f.dofilterAdaptive(data[i], sinusoid, lR)

plt.plot(t, w)
plt.title('ECG 50Hz LMS Filter')
plt.xlabel('Time (sec)')
plt.ylabel('Normalized ECG (volts)')

plt.show()