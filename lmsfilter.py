import numpy as np
import matplotlib.pyplot as plt
# from hpbsfilter import fir
import AdaptFilter
import firfilter

fs = 250
taps = fs * 2
f0 = 50
data = np.loadtxt("ECG_msc_matric_5.dat")
lR = 0.0001
t_max = len(data) * 20

coeff = np.zeros(taps)
w = np.empty(len(data))
# dc = np.random.normal(0,1,len(data))

# f = AdaptFilter.lmsFilter(fs, coeff)
f = firfilter.firFilter(fs,coeff)
for i in range(len(data)):
    sinusoid = (np.sin(2 * np.pi * i * (f0 / fs)))
    w[i] = f.dofilterAdaptive(data[i], sinusoid, lR)
    # w[i] = f.dofilterAdaptive(data[i], sinusoid + dc[i], lR)

t2 = np.linspace(0, t_max, len(w))
# plt.subplot(1, 2, 1)
plt.plot(t2, w)
plt.title('ECG 50Hz LMS Filter')
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

# t1 = np.linspace(0, len(fir) * 20, len(fir))
#
# plt.subplot(1, 2, 2)
# plt.plot(t1, fir)
# plt.xlim(0, t_max + 5000)
# plt.title('ECG 50Hz and Dc Noise Removed')
# plt.xlabel('time(sec)')
# plt.ylabel('ECG (volts)')

plt.show()

