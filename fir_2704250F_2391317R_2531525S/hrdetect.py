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


"""Pull out one ECG action"""


def get_ecgaction(dataset, dataset_time, start, stop):
    data_range = dataset[start:stop]  # define the data points of interest
    time_range = dataset_time[start:stop]  # define the time range in which those data points occur

    return data_range, time_range  # return both data and time arrays of the ecg action


"""Create a sinc wavelet"""


def get_wavelet(length, time_reversed_dataset, time_range):
    data_val = max(time_reversed_dataset) * np.sinc(length * 25)  # define the characteristics of the sinc wavelet
    data_val[0:int(len(time_range) / 2) - 8] = 0  # zero out all values of no interest to
    # the left of the function's peak
    data_val[int(len(time_range) / 2) + 8:len(time_range)] = 0  # zero out all values of no interest to
    # the right of the function's peak

    return data_val  # return the wavelet data array


"""R Peak Threshold Function"""


def threshold(dataset):
    val = max(dataset[700:])  # 700 was picked as we want to avoid the anomalies caused by the filter starting up
    highest_volt = val + val / 2  # Dynamically set the max of the threshold
    lowest_volt = val * 0.5  # Dynamically set the min of the threshold

    return lowest_volt, highest_volt  # return the max and min threshold of the dataset


"""Generate a list to store peak times"""


def get_peaktime(dataset, upper, lower, r):
    data_points = []  # create a list to store data points
    iter_val = 0
    while iter_val < (len(dataset)):
        if upper > dataset[iter_val] > lower:  # set the condition for data storage
            data_points.append(r[iter_val])  # store data points
            iter_val += 50  # we add 50 data points once we find our max point to avoid the next closest peak from
            # overwriting our peak value
        else:
            iter_val += 1  # we add 1 data point to move to the next iteration value

    return data_points  # return the list of data points


"""Generate a list to store bpm_fir_wavelet values"""


def get_bpm(dataset):
    data_points = []  # create a list to store data points

    for iter_val in range(len(dataset) - 1):
        data_points.append(60 / (dataset[iter_val + 1] - dataset[iter_val]))  # perform the bpm calculation

    return data_points  # return the list of data points


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

# Q4

"""Find the range in the FIR plot where an ECG action occurs and plot it"""

plt.figure(1)
plt.subplot(1, 2, 1)
template, ecgaction_time = get_ecgaction(fir, t_data, 950, 1200)  # call the function to pull out one ecg action
plt.plot(ecgaction_time, template)
plt.title("matched filter template")
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

"""Plot the time reversed version of the template """

plt.subplot(1, 2, 2)
time_reversed_coeff = template[::-1]  # time reverse the template to obtain desired coefficient values
plt.plot(ecgaction_time, time_reversed_coeff, label='Time reversed')
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

"""Create and plot the sinc function"""

n_coeff = get_wavelet(np.linspace(-1, 1, len(ecgaction_time)), time_reversed_coeff, ecgaction_time)
plt.subplot(1, 2, 2)
plt.plot(ecgaction_time, n_coeff, label='Sinc function')
plt.title("Time reversed match filter and sinc function")
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

ecg_time = np.linspace(0, t_max, len(fir))

plt.figure(2)
plt.subplot(1, 2, 1)
plt.plot(ecg_time, fir)
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')
plt.title('Original FIR output')

plt.subplot(1, 2, 2)
plt.plot(ecg_time, fir_wavelet)
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')
plt.title('Sinc function on original FIR')

"""Define and plot the R peak threshold """

plt.figure(3)
min_thresh, max_thresh = threshold(fir_wavelet)
plt.plot(ecg_time, fir_wavelet)
plt.xlim(2.5)  # Limit the x-axis to start from 2.5 since we don't_data want the range of values at which our filter
# starts
plt.ylim(min_thresh, max_thresh)  # Limit the y-axis between max threshold and min threshold values
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')
plt.title('Threshold R-Peaks plot')

"""Call the get peak_time function to create a list of peak times for the wavelet influenced FIR"""
peak_time_fir_wavelet = get_peaktime(fir_wavelet, max_thresh, min_thresh, ecg_time)

"""Call the get bpm function to create a list of bpm values for the wavelet influenced FIR"""
bpm_fir_wavelet = get_bpm(peak_time_fir_wavelet)

"""Define the original FIR R peak threshold """
min_FIR_thresh, max_FIR_thresh = threshold(fir)

"""Call the get peak_time function to create a list of peak times for the original FIR"""
peak_time_fir = get_peaktime(fir, max_FIR_thresh, min_FIR_thresh, ecg_time)

"""Call the get bpm function to create a list of bpm values for the original FIR"""
bpm_fir = get_bpm(peak_time_fir)

"""Plot the momentary heart rate for the original fir and the fir_wavelet"""

plt.figure(4)
plt.step(peak_time_fir[2:], bpm_fir[1:], label="Original FIR")
plt.step(peak_time_fir_wavelet[2:], bpm_fir_wavelet[1:], label="Wavelet Influenced")
plt.xlabel('time(sec)')
plt.ylabel('BPM')
plt.title('Momentary Heart Rate')
plt.legend(loc="upper right")

plt.show()
