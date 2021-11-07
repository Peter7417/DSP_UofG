import numpy as np
import matplotlib.pyplot as plt
import firfilter

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
"""Plot the ECG"""
data = np.loadtxt('ECG_msc_matric_5.dat')
t_max = 20
t = np.linspace(0, t_max, len(data))

fs = 250  # sample frequency
taps = (fs * 2)  # defining taps

"""Bandstop"""
f1 = int((45 / fs) * taps)  # before 50Hz
f2 = int((55 / fs) * taps)  # after 50Hz

"""Highpass"""
f3 = int((0.5 / fs) * taps)  # ideal for cutting off DC noise

"""Call the function for Bandstop and Highpass"""
impulse_BS = bandstopDesign(fs, f1, f2)
impulse_HP = highpassDesign(fs, f3)

"""Reshuffle the coefficients for highpass"""
h = np.zeros(taps)
h[0:int(taps / 2)] = impulse_HP[int(taps / 2):taps]
h[int(taps / 2):taps] = impulse_HP[0:int(taps / 2)]
h_new = h * np.hanning(taps)

"""Reshuffle the coefficients for bandstop"""

h1 = np.zeros(taps)
h1[0:int(taps / 2)] = impulse_BS[int(taps / 2):taps]
h1[int(taps / 2):taps] = impulse_BS[0:int(taps / 2)]
h_new1 = h1 * np.hanning(taps)

"""Call the class method dofilter, by passing in only a scalar value at a time which outputs a scalar value"""
fir_HP = np.empty(len(data))
fi = firfilter.firFilter(h_new)
for i in range(len(fir_HP)):
    fir_HP[i] = fi.dofilter(data[i])

fir = np.empty(len(data))
po = firfilter.firFilter(h_new1)
for i in range(len(fir)):
    fir[i] = po.dofilter(fir_HP[i])


# Q4

"""Find the range in the FIR plot where a heart beat occurs and plot it"""

"Test"

plt.figure(1)
plt.subplot(1, 2, 1)
template = fir[950:1200]
time = t[950:1200]
plt.plot(time, template)
plt.title("matched filter template")
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

"""Plot the time reversed version of the template """
plt.subplot(1, 2, 2)
coefficients = template[::-1]
plt.plot(time, coefficients)
plt.title("matched filter time reversed + sinc func")
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

"""Create the sinc function"""
wavelet = np.linspace(-1, 1, len(time))
plt.subplot(1, 2, 2)
n_coeff = max(coefficients) * np.sinc(wavelet * 25)
n_coeff[0:int(len(time) / 2) - 8] = 0
n_coeff[int(len(time) / 2) + 8:len(time)] = 0
plt.plot(time, n_coeff)
n_coeff = n_coeff ** 5  # Raised to the power of 5 to show the significant difference between the highest peak and the
# smallest peak

# Subplot adjustments
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

""""Use the dofilter function with the wavelet and the previous data set to find the new fir data set"""
fir1 = np.empty(len(data))
fi = firfilter.firFilter(n_coeff)
for i in range(len(fir1)):
    fir1[i] = fi.dofilter(fir[i])

"""Plot both the original FIR and the new FIR"""
t1 = np.linspace(0, t_max, len(fir))
t2 = np.linspace(0, t_max, len(fir1))
plt.figure(2)
plt.subplot(1, 2, 1)
plt.plot(t1, fir)
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')
plt.title('Original FIR output')
plt.subplot(1, 2, 2)
plt.plot(t2, fir1)
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')
plt.title('Sinc function on original FIR')

"""Define the R peak threshold """
plt.figure(3)
val = max(fir1[700:])  # 700 was picked since we want to avoid the anomalies caused by the filter starting up
max_t = val + val / 2  # Dynamically set the max of the threshold
min_t = val * 0.5   # Dynamically set the min of the threshold
plt.plot(t2, fir1)
plt.xlim(2.5)   # Limit the x-axis to start from 2.5
plt.ylim(min_t, max_t)  # Limit the y-axis between max threshold and min threshold
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')
plt.title('Threshold R-Peaks plot')

"""Create a list to store the time values where fir1 peaks"""
peak_time = []
i = 0
while i < (len(fir1)):
    if max_t > fir1[i] > min_t:
        peak_time.append(t2[i])
        i += 50
    else:
        i += 1

"""Create a list to store the BPM values in reference to the peak times"""
bpm = []

for i in range(len(peak_time) - 1):
    bpm.append(60 / (peak_time[i + 1] - peak_time[i]))

"""Plot the momentary heart rate"""
plt.figure(4)
plt.step(peak_time[2:], bpm[1:], label="Wavelet Influenced")
plt.xlabel('time(sec)')
plt.ylabel('BPM')
plt.title('Momentary Heart Rate')


"""Define the original FIR R peak threshold """
val = max(fir[700:])  # 700 was picked since we want to avoid the anomalies caused by the filter starting up
max_t1 = val + val / 2  # Dynamically set the max of the threshold
min_t1 = val * 0.5   # Dynamically set the min of the threshold

"""Create a list to store the time values where fir peaks"""
peak_time_1 = []
i = 0
while i < (len(fir)):
    if max_t1 > fir[i] > min_t1:
        peak_time_1.append(t1[i])
        i += 50
    else:
        i += 1

"""Create a list to store the BPM values in reference to the peak times"""
bpm_1 = []

for i in range(len(peak_time_1) - 1):
    bpm_1.append(60 / (peak_time_1[i + 1] - peak_time_1[i]))


"""Plot the momentary heart rate"""
plt.figure(4)
plt.step(peak_time_1[2:], bpm_1[1:], label="Original FIR")
plt.xlabel('time(sec)')
plt.ylabel('BPM')
plt.title('Momentary Heart Rate')
plt.legend(loc="upper right")
plt.show()

