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
    ls[0] = x[0] * y[0]
    return np.array(ls)


def convolveValid(x, y):
    ls = []  # Create a list to store the convolution array in
    l1 = len(x)  # Find the length of the data 1 set
    l2 = len(y)  # FInd the length of the data 2 set
    N = max(l2, l1) - min(l2, l1) + 1  # Find the length of the convolved array
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
    ls[0] = x[0] * y[0]
    return np.array(ls)


"""50 HZ removal"""


def bandstopDesign(w1, w2, M):
    # frequency resolution =0.5
    cutoff_1 = w1
    cutoff_2 = w2
    X = np.ones(M)
    X[cutoff_1:cutoff_2 + 1] = 0
    X[M - cutoff_2:M - cutoff_1 + 1] = 0
    x = np.real(np.fft.ifft(X))

    return x


"""DC noise removal"""


def highpassDesign(w3, M):
    # frequency resolution =0.5
    cutoff = w3
    X = np.ones(M)
    X[0:cutoff + 1] = 0
    X[M - cutoff: M + 1] = 0
    x = np.real(np.fft.ifft(X))

    return x


# Q1 and Q2
"""Plot the ECG"""
# dataRaw = np.loadtxt('ECG_msc_matric_4.dat')
# data = dataRaw / max(dataRaw)
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
impulse_BS = bandstopDesign(f1, f2, taps)
impulse_HP = highpassDesign(f3, taps)

"""Reshuffle the coefficients for highpass"""
h_HP = np.zeros(taps)
h_HP[0:int(taps / 2)] = impulse_HP[int(taps / 2):taps]
h_HP[int(taps / 2):taps] = impulse_HP[0:int(taps / 2)]
h_newHP = h_HP * np.hanning(taps)

"""Reshuffle the coefficients for bandstop"""
h_BS = np.zeros(taps)
h_BS[0:int(taps / 2)] = impulse_BS[int(taps / 2):taps]
h_BS[int(taps / 2):taps] = impulse_BS[0:int(taps / 2)]
h_newBS = h_BS * np.hanning(taps)

"""Call the class method dofilter, by passing in only a scalar value at a time which outputs a scalar value"""
fir_HP = np.empty(len(data))
fi = firfilter.firFilter(h_newHP)
for i in range(len(fir_HP)):
    fir_HP[i] = fi.dofilter(data[i])

fir = np.empty(len(data))
po = firfilter.firFilter(h_newBS)
for i in range(len(fir)):
    fir[i] = po.dofilter(fir_HP[i])

"""Plot both the original ECG data set and new filtered data set """
plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(t, data)
plt.title('ECG')
plt.xlabel('time(sec)')
plt.ylabel('Normalized ECG (volts)')

t1 = np.linspace(0, t_max, len(fir))
plt.subplot(1, 2, 2)
plt.plot(t1, fir)
plt.xlim(0, t_max)
plt.title('ECG 50Hz and Dc Noise Removed')
plt.xlabel('time(sec)')
plt.ylabel('Normalized ECG (volts)')


plt.show()
