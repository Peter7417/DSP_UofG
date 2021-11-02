import numpy as np
import matplotlib.pyplot as plt
import firfilter

# Q3
fs = 250
taps = fs * 2
f0 = 50
# dataRaw = np.loadtxt('ECG_msc_matric_4.dat')
# data = dataRaw / max(dataRaw)
data = np.loadtxt('ECG_msc_matric_5.dat')
lR = 0.00089
t_max = 20


coeff = np.zeros(taps)
w = np.empty(len(data))


f = firfilter.firFilter(coeff)
for i in range(len(data)):
    sinusoid = (np.sin(2 * np.pi * i * (f0 / fs)))
    w[i] = f.dofilterAdaptive(data[i], sinusoid, lR)

t2 = np.linspace(0, t_max, len(w))
plt.plot(t2, w)
plt.title('ECG 50Hz LMS Filter')
plt.xlabel('time(sec)')
plt.ylabel('Normalized ECG (volts)')


plt.show()

