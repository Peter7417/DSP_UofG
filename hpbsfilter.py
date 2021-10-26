import numpy as np
import matplotlib.pyplot as plt
import firfilter


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
fil = firfilter.firFilter(fs, coeff, 0)
for i in range(len(h)):
    k, p = fil.getImpulse(coeff[i])
    h[p] = k

h_new = h * np.blackman(taps)

"""Call the class method dofilter, by passing in only a scalar value which outputs a scalar value, 
which performs a simple multiplication operation to complete the convolution process"""
N = len(h_new) + len(data) - 1
l1 = len(h_new)
l2 = len(data)
lst = []
b = np.zeros(len(h_new) + len(data) - 1)

for n in range(N):
    for i in range(l1):
        if 0 <= (n - i + 1) < l2:
            b[n] = b[n] + firfilter.firFilter(fs, h_new, i).dofilter(data[n-i+1])

for i in b:
    lst.append(i)

"""Rearrange the list to move the last value to the front and replace it with the first value of data set 1
and convert back to an array"""
lst.insert(0, lst.pop())
lst[0] = h_new[0]
fir = np.array(lst)

"""Plot both the original ECG data set and new filtered data set """
plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(t, data)
plt.title('ECG')
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

t1 = np.linspace(0, t_max, len(fir))
plt.subplot(1, 2, 2)
plt.plot(t1, fir)
plt.xlim(0, t_max)
plt.title('ECG 50Hz and Dc Noise Removed')
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

plt.show()
