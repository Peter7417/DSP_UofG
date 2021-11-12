import numpy as np
import matplotlib.pyplot as plt
import firfilter

"""Reshuffle Function"""


def reshuffle(filter_coeff):
    h = np.zeros(taps)
    h[0:int(taps / 2)] = filter_coeff[int(taps / 2):taps]
    h[int(taps / 2):taps] = filter_coeff[0:int(taps / 2)]
    return h * np.hanning(taps)


"""Pull out one ECG action"""


def get_ecgaction(dataset, dataset_time, start, stop):
    data_range = dataset[start:stop]
    time_range = dataset_time[start:stop]

    return data_range, time_range


"""Create a sinc wavelet"""


def get_wavelet(length):
    data_val = max(coefficients) * np.sinc(length * 25)
    data_val[0:int(len(time) / 2) - 8] = 0
    data_val[int(len(time) / 2) + 8:len(time)] = 0
    return data_val


"""R Peak Threshold Function"""


def threshold(dataset):
    val = max(dataset[700:])  # 700 was picked since we want to avoid the anomalies caused by the filter starting up
    highest_volt = val + val / 2  # Dynamically set the max of the threshold
    lowest_volt = val * 0.5  # Dynamically set the min of the threshold

    return lowest_volt, highest_volt


"""Generate a list to store peak times"""


def get_peaktime(dataset, upper, lower, r):
    data_points = []
    iter_val = 0
    while iter_val < (len(dataset)):
        if upper > dataset[iter_val] > lower:
            data_points.append(r[iter_val])
            iter_val += 50
        else:
            iter_val += 1

    return data_points


"""Generate a list to store bpm_fir_wavelet values"""


def get_bpm(dataset):
    data_points = []

    for iter_val in range(len(dataset) - 1):
        data_points.append(60 / (dataset[iter_val + 1] - dataset[iter_val]))

    return data_points


"""50 HZ removal"""


def bandstopDesign(samplerate, w1, w2):
    # frequency resolution =0.5
    M = samplerate * 2
    X = np.ones(M)
    X[w1:w2 + 1] = 0
    X[M - w2:M - w1 + 1] = 0
    x = np.real(np.fft.ifft(X))

    return x


"""DC noise removal"""


def highpassDesign(samplerate, w3):
    # frequency resolution =0.5
    M = samplerate * 2
    X = np.ones(M)
    X[0:w3 + 1] = 0
    X[M - w3: M + 1] = 0
    x = np.real(np.fft.ifft(X))

    return x


"""Load data into python"""
data = np.loadtxt('ECG_msc_matric_5.dat')

"""Define constants"""
fs = 250  # sample frequency
t_max = len(data) / fs  # sample time of data
t_data = np.linspace(0, t_max, len(data))  # create an array to model the x-axis with time values
taps = (fs * 2)  # defining taps

"""Bandstop"""
f1 = int((45 / fs) * taps)  # cutoff frequency before 50Hz
f2 = int((55 / fs) * taps)  # cutoff frequency after 50Hz

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

# Q4

"""Find the range in the FIR plot where an ECG action occurs and plot it"""

plt.figure(1)
plt.subplot(1, 2, 1)
template, time = get_ecgaction(fir, t_data, 950, 1200)  # call the function to pull out one ecg action
plt.plot(time, template)
plt.title("matched filter template")
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

"""Plot the time reversed version of the template """

plt.subplot(1, 2, 2)
coefficients = template[::-1]  # time reverse the template to obtain desired coefficient values
plt.plot(time, coefficients, label='Time reversed')
plt.title("matched filter time reversed + sinc func")
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

"""Create and plot the sinc function"""

n_coeff = get_wavelet(np.linspace(-1, 1, len(time)))
plt.subplot(1, 2, 2)
plt.plot(time, n_coeff, label='Sinc function')
plt.legend(loc='upper right')
n_coeff = n_coeff ** 5  # Raised to the power of 5 to show the significant difference between the highest peak and the
# smallest peak


""""Call the dofilter function in the FIR class with the wavelet data 
and the filtered FIR data set to find the new fir data set influenced by a wavelet"""

fir_wavelet = np.empty(len(data))
fi = firfilter.firFilter(n_coeff)
for i in range(len(fir_wavelet)):
    fir_wavelet[i] = fi.dofilter(fir[i])

"""Plot both the original FIR and the new FIR"""

fir_time = np.linspace(0, t_max, len(fir))
fir_wavelet_time = np.linspace(0, t_max, len(fir_wavelet))
plt.figure(2)
plt.subplot(1, 2, 1)
plt.plot(fir_time, fir)
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')
plt.title('Original FIR output')

plt.subplot(1, 2, 2)
plt.plot(fir_wavelet_time, fir_wavelet)
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')
plt.title('Sinc function on original FIR')

"""Define the R peak threshold """
plt.figure(3)
min_thresh, max_thresh = threshold(fir_wavelet)
plt.plot(fir_wavelet_time, fir_wavelet)
plt.xlim(
    2.5)  # Limit the x-axis to start from 2.5 since we don't_data want the range of values at which our filter starts
plt.ylim(min_thresh, max_thresh)  # Limit the y-axis between max threshold and min threshold values
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')
plt.title('Threshold R-Peaks plot')

"""Call the get peak time function to create a list of peak times for the wavelet influenced FIR"""
peak_time_fir_wavelet = get_peaktime(fir_wavelet, max_thresh, min_thresh, fir_wavelet_time)

"""Call the get bpm function to create a list of bpm values for the wavelet influenced FIR"""
bpm_fir_wavelet = get_bpm(peak_time_fir_wavelet)

"""Define the original FIR R peak threshold """
min_FIR_thresh, max_FIR_thresh = threshold(fir)

"""Call the get peak time function to create a list of peak times for the original FIR"""
peak_time_fir = get_peaktime(fir, max_FIR_thresh, min_FIR_thresh, fir_time)

"""Call the get bpm function to create a list of bpm values for the original FIR"""
bpm_fir = get_bpm(peak_time_fir)

"""Plot the momentary heart rate for the original fir"""
plt.figure(4)
plt.step(peak_time_fir[2:], bpm_fir[1:], label="Original FIR")
plt.xlabel('time(sec)')
plt.ylabel('BPM')
plt.title('Momentary Heart Rate')

"""Plot the momentary heart rate for fir_wavelet"""
plt.figure(4)
plt.step(peak_time_fir_wavelet[2:], bpm_fir_wavelet[1:], label="Wavelet Influenced")
plt.xlabel('time(sec)')
plt.ylabel('BPM')
plt.title('Momentary Heart Rate')
plt.legend(loc="upper right")

plt.show()
