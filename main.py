import numpy as np
import matplotlib.pyplot as plt
import filter

"""Create a function to perform convolution """


def convolve(x, y):
    ls = []  # Create a list to store the convolution array in
    l1 = len(x)  # Find the length of the data 1 set
    l2 = len(y)  # FInd the length of the data 2 set
    N = l1 + l2 - 1  # Find the length of the convolved array
    k = np.zeros(N)  # Create an array of zeros to store the convolution result

    """Logic behind the convolution operation"""
    for n in range(N):
        for p in range(l1):
            if 0 <= (n - p + 1) < l2:
                k[n] = k[n] + x[p] * y[n - p + 1]

    """Append all convolution values in a list"""
    for j in k:
        ls.append(j)

    """Rearrange the list to move the last value to the front and replace it with the first value of data set 1 
    and convert back to an array"""
    ls.insert(0, ls.pop())
    ls[0] = x[0]
    return np.array(ls)


"""50 HZ removal"""


def bandstopDesign(freq, w1, w2):
    taps = freq * 2  # frequency resolution =0.5
    cutoff_1 = int(w1 * taps)
    cutoff_2 = int(w2 * taps)
    X = np.ones(taps)
    X[cutoff_1:cutoff_2 + 1] = 0
    X[taps - cutoff_2:taps - cutoff_1 + 1] = 0
    x = np.real(np.fft.ifft(X))

    return x


"""DC noise removal"""


def highpassDesign(freq, w2):
    taps = freq * 2  # frequency resolution =0.5
    cutoff = int(w2 * taps)
    X = np.ones(taps)
    X[0:cutoff + 1] = 0
    x = np.real(np.fft.ifft(X))

    return x


# Q1 and Q2
"""Plot the ECG"""
data = np.loadtxt('ECG_msc_matric_5.dat')
t_max = len(data) * 20
t = np.linspace(0, t_max, len(data))

f0 = 50  # noise frequency
lR = 0.0001  # learning rate
fs = 250  # sample frequency

"""Bandstop"""
f1 = 45 / fs  # before 50Hz
f2 = 55 / fs  # after 50Hz

"""Highpass"""
f3 = 0.5 / fs  # ideal for cutting off DC noise

"""Call the function for Bandstop and Highpass"""
impulse_BS = bandstopDesign(fs, f1, f2)
impulse_HP = highpassDesign(fs, f3)

"""Convolve the coefficients of both the Bandstop and Highpass"""
coeff = convolve(impulse_HP, impulse_BS)

"""Call the class to get the reshuffled impulse response by feeding in data one at a time"""
fil = filter.firFilter(fs, coeff)
for i in range(len(coeff)):
    h_new = fil.dofilter(coeff[i])

"""Convolve the new impulse response with the ecg data"""
fir = convolve(h_new, data)

"""Plot both the original ECG data set and new filtered data set """
plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(t, data)
plt.title('ECG')
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

t1 = np.linspace(0, len(fir) * 20, len(fir))
plt.subplot(1, 2, 2)
plt.plot(t1, fir)
plt.xlim(0, t_max + 5000)
plt.title('ECG 50Hz and Dc Noise Removed')
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

# Q3

"""Create an empty data set to store the output in and 
another data set to hold the changing coefficients of the LMS filter"""
w = np.empty(len(data))
coeff = np.zeros(fs * 2)

"""Call the class function and pass in one data point at a time to give us the output of the lms filter"""
f = filter.firFilter(fs, coeff)
for i in range(len(data)):
    sinusoid = (np.sin(2 * np.pi * i * (f0 / fs)))
    w[i] = f.dofilterAdaptive(data[i], sinusoid, lR)

"""Plot the FIR data set and the LMS data set"""
t2 = np.linspace(0, t_max, len(w))
plt.figure(2)
plt.subplot(1, 2, 1)
plt.plot(t1, fir)
plt.xlim(0, t_max + 5000)
plt.title('ECG 50Hz FIR Filter')
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

plt.subplot(1, 2, 2)
plt.plot(t2, w)
plt.title('ECG 50Hz LMS Filter')
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

# Q4

"""Find the range in the FIR plot where a heart beat occurs and plot it"""
plt.figure(3)
plt.subplot(1, 2, 1)
template = fir[1200:1600]
plt.plot(template)
plt.title("matched filter template")
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

"""Plot the time reversed version if the template """
plt.subplot(1, 2, 2)
coefficients = template[::-1]
plt.plot(coefficients)
plt.title("matched filter time reversed")
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

# coefficients = coefficients**2
# plt.figure(4)
# plt.plot(signal.lfilter(coefficients,1,data))


plt.show()
