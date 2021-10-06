import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os


# Functions

def get_arrayIndex(array, fs, frequency):
    return int(len(array) / fs * frequency)


def get_noiseReduction(array,noise_factor):
    return array / noise_factor


def get_amplification(array, amp_factor):
    return array * amp_factor


# Question 1

"""Import an audio file in .wav format at 48KHz"""
path = os.getcwd()
file_name = 'FFT Test.wav'
# file_name = 'FFT_2.wav'

location = os.path.join(path, file_name)
samplerate, data = wavfile.read(location)


"""Create the FFT of the data(np array of numbers) obtained from the audio file"""
data_fft = np.fft.fft(data)


"""Create an array of half range of values from the fourier transform so we can plot up to the nyquist frequency"""
half_rangeValues = np.abs(data_fft[0:int(len(data_fft) / 2 - 1)])

"""Define the frequency and time domains"""
f = np.linspace(0, samplerate / 2, len(half_rangeValues))
t = np.linspace(0, len(data) / samplerate, len(data))

"""Plot the Time domain spectrum"""
# plt.subplot(1, 2, 1)
# plt.plot(t, data / ((2**16)/2))
plt.title('Time Domain')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')

print(half_rangeValues[0])

"""Plot the Frequency domain spectrum"""
dB = 20*np.log10(abs(half_rangeValues) / max(half_rangeValues))
# plt.subplot(1, 2, 2)
plt.figure(1)
plt.plot(f, dB)
plt.xscale('log')
plt.title('Frequency Domain')
plt.xlabel('Frequency(rad/s)')
plt.ylabel('Amplitude(dB)')

print(max(half_rangeValues))

# plt.show()


