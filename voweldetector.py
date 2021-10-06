import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks


a = 710, 1100, 2450
I = 280, 2250, 2890
u = 400, 1920, 2650

"""Import an audio file in .wav format at 48KHz"""
path = os.getcwd()
file_name = 'FFT Test.wav'
# file_name = 'FFT_2.wav'

location = os.path.join(path, file_name)
samplerate, data = wavfile.read(location)

"""Create the FFT of the data(np array of numbers) obtained from the audio file"""
data_fft = np.fft.fft(data)

"""Create an array of half range of values from the fourier transform so we can plot up to the nyquist frequency"""
half_rangeValues = data_fft[0:int(len(data_fft) / 2 - 1)]

"""Define the frequency and time domains"""
f = np.linspace(0, samplerate / 2, len(half_rangeValues))

"""Plot the Frequency domain spectrum"""
dB = 20 * np.log10(half_rangeValues / len(data_fft))
plt.subplot(1,2,1)
plt.plot(f, dB)
plt.xscale('log')
plt.title('Frequency Domain')
plt.xlabel('Frequency(rad/s)')
plt.ylabel('Amplitude(dB)')

peaks, _ = find_peaks(dB, height=1, threshold=1, distance=1)
plt.subplot(1, 2, 2)
plt.xscale('log')
plt.plot(dB)
plt.plot(peaks, dB[peaks])
plt.title('Peaks Comparison')
plt.xlabel('Peak Index')
plt.ylabel('Amplitude(dB)')
plt.show()


for i in peaks:
    if i in a:
        print('a')
    elif i in I:
        print('i')
    elif i in u:
        print('u')

