import numpy as np
import matplotlib.pyplot as plt
import firfilter
import AdaptFilter

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
t_max = 20
t = np.linspace(0, t_max, len(data))

f0 = 50  # noise frequency
lR = 0.0001  # learning rate
fs = 250  # sample frequency
taps = fs * 2

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
h = np.zeros(taps)
fil = AdaptFilter.firFilter(fs, coeff)
for i in range(len(h)):
    k, p = fil.dofilter(coeff[i])
    h[p] = k

h_new = h * np.blackman(taps)
# print(h_new)
"""Convolve the new impulse response with the ecg data"""
fir = convolve(h_new, data)

plt.figure(1)
plt.plot(fir)

"""Find the range in the FIR plot where a heart beat occurs and plot it"""
plt.figure(2)
plt.subplot(1, 2, 1)
template = fir[1200:1600]
time = t[1200:1600]
plt.plot(time,template)
plt.title("matched filter template")
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

"""Plot the time reversed version if the template """
plt.subplot(1, 2, 2)
coefficients = template[::-1]
plt.plot(time,coefficients)
plt.title("matched filter time reversed")
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

plt.show()
