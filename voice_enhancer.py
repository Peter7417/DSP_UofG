import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# Constants:

ns_1 = 10**8
ns_2 = 50**8
amp_1 = 5
amp_2 = 10
amp_3 = 25
amp_4 = 8
max_values = 2 ** 16


# Functions

def get_arrayIndex(array, fs, frequency):
    return int(len(array) / fs * frequency)


def get_noiseReduction(array, noise_factor):
    return array / noise_factor


def get_amplification(array, amp_factor):
    return array * amp_factor


# Question 1

"""Import an audio file in .wav format at 48KHz"""
path = os.getcwd()
file_name = 'FFT Test.wav'
# file_name = 'FFT_2.wav'
# file_name = 'New FFT.wav'

location = os.path.join(path, file_name)
samplerate, data = wavfile.read(location)

"""Create the FFT of the data(np array of numbers) obtained from the audio file"""
data_fft = np.fft.fft(data)

"""Create an array of half range of values from the fourier transform so we can plot up to the nyquist frequency"""
half_rangeValues = data_fft[0:int(len(data_fft) / 2 - 1)]

"""Define the frequency and time domains"""
f = np.linspace(0, samplerate / 2, len(half_rangeValues))
t = np.linspace(0, len(data) / samplerate, len(data))

"""Plot the Time domain spectrum"""
plt.subplot(2, 2, 1)
plt.plot(t, data / (max_values / 2))
plt.title('Time Domain')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')

"""Plot the Frequency domain spectrum"""
dB = 20 * np.log10(half_rangeValues / max(half_rangeValues))    # Using dB = 20log(A/A_0)
plt.subplot(2, 2, 2)
plt.plot(f, dB)
plt.xscale('log')
plt.title('Frequency Domain')
plt.xlabel('Frequency(rad/s)')
plt.ylabel('Amplitude(dB)')

# vowels = set("AEIOUaeiou")
# cons = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")

# Question 3

# Amplification and noise reduction

"""Values for lower frequency noise at 0 - 100Hz """
noise_1 = get_arrayIndex(data_fft, samplerate, 0)  # array location at 0Hz
noise_2 = get_arrayIndex(data_fft, samplerate, 99)  # array location at 99Hz

"""Values for the harmonic frequency at 1KHz to 10Khz"""
k1 = get_arrayIndex(data_fft, samplerate, 1000)  # array location at 1000Hz
k2 = get_arrayIndex(data_fft, samplerate, 2500)  # array location at 2500Hz

k3 = get_arrayIndex(data_fft, samplerate, 2501)  # array location at 2501Hz
k4 = get_arrayIndex(data_fft, samplerate, 5000)  # array location at 5000Hz

k5 = get_arrayIndex(data_fft, samplerate, 5001)  # array location at 5001Hz
k6 = get_arrayIndex(data_fft, samplerate, 7500)  # array location at 7500Hz

k7 = get_arrayIndex(data_fft, samplerate, 7501)  # array location at 7501Hz
k8 = get_arrayIndex(data_fft, samplerate, 10000)  # array location at 10000Hz

"""Values for the higher frequency noise at 10KHz - 24KHz"""
noise_3 = get_arrayIndex(data_fft, samplerate, 10001)  # array location at 10001Hz
noise_4 = get_arrayIndex(data_fft, samplerate, 24000)  # array location at 24000Hz

"""Perform noise reduction by a factor so the resulting DB is almost negated. Call the get_noiseReduction function on 
both the area before N/2 and after to obtain a mirroring effect"""
data_fft[noise_1:noise_2] = get_noiseReduction(data_fft[noise_1:noise_2], ns_1)  # Lower frequency noise reduction
data_fft[int(len(data_fft) - noise_2):int(len(data_fft) - noise_1)] = \
    get_noiseReduction(data_fft[int(len(data_fft) - noise_2):int(len(data_fft) - noise_1)], ns_1)  # Mirroring

data_fft[noise_3:noise_4] = get_noiseReduction(data_fft[noise_3:noise_4], ns_2)  # Higher frequency noise reduction
data_fft[int(len(data_fft) - noise_4):int(len(data_fft) - noise_3)] = \
    get_noiseReduction(data_fft[int(len(data_fft) - noise_4):int(len(data_fft) - noise_3)], ns_2)  # Mirroring

"""Perform amplification by a chosen factor so the resulting audio file is clearer than the original. 
Call the get_amplification function on both the area before N/2 and after to obtain a mirroring effect"""
data_fft[k1:k2] = get_amplification(data_fft[k1:k2], amp_1)  # Harmonics amplification
data_fft[int(len(data_fft) - k2):int(len(data_fft) - k1)] = \
    get_amplification(data_fft[int(len(data_fft) - k2):int(len(data_fft) - k1)], amp_1)  # Mirroring

data_fft[k3:k4] = get_amplification(data_fft[k3:k4], amp_2)  # Harmonics amplification
data_fft[int(len(data_fft) - k4):int(len(data_fft) - k3)] = \
    get_amplification(data_fft[int(len(data_fft) - k4):int(len(data_fft) - k3)], amp_2)  # Mirroring

data_fft[k5:k6] = get_amplification(data_fft[k5:k6], amp_3)  # Harmonics amplification
data_fft[int(len(data_fft) - k6):int(len(data_fft) - k5)] = \
    get_amplification(data_fft[int(len(data_fft) - k6):int(len(data_fft) - k5)], amp_3)  # Mirroring

data_fft[k7:k8] = get_amplification(data_fft[k7:k8], amp_4)  # Harmonics amplification
data_fft[int(len(data_fft) - k7):int(len(data_fft) - k8)] = \
    get_amplification(data_fft[int(len(data_fft) - k7):int(len(data_fft) - k8)], amp_4)  # Mirroring

"""Plot the improved fft"""
new_halfRange = data_fft[0:int(len(data_fft) / 2 - 1)]
new_dB = 20 * np.log10(new_halfRange / max(half_rangeValues))
plt.subplot(2, 2, 4)
plt.plot(f, new_dB)
plt.xscale('log')
plt.title('Improved Frequency Domain')
plt.xlabel('Frequency(rad/s)')
plt.ylabel('Amplitude(dB)')

"""Plot the improved sound wave by performing inverse fourier transform"""
raw_data = np.fft.ifft(data_fft)
new_data = np.real(raw_data)
wav_file = new_data.astype(np.int16)
plt.subplot(2, 2, 3)
plt.plot(t, new_data / (max_values / 2))
plt.title('Improved Time Domain')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')


"""Create a new audio file that stores the improved version of the initial audio file"""
name = str('Improved ' + file_name)
wavfile.write(name, samplerate, wav_file)

# Subplot adjustments
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

plt.show()
