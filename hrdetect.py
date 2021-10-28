import numpy as np
import matplotlib.pyplot as plt
import firfilter


"""Create a function to perform convolution """


def convolveFull(x, y):
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
    ls[0] = x[0]*y[0]
    return np.array(ls)


"""50 HZ removal"""


def bandstopDesign(w1, w2, M):
    # frequency resolution =0.5
    cutoff_1 = int(w1 * M)
    cutoff_2 = int(w2 * M)
    X = np.ones(M)
    X[cutoff_1:cutoff_2 + 1] = 0
    X[M - cutoff_2:M - cutoff_1 + 1] = 0
    x = np.real(np.fft.ifft(X))

    return x


"""DC noise removal"""


def highpassDesign(w2, M):
    # frequency resolution =0.5
    cutoff = int(w2 * M)
    X = np.ones(M)
    X[0:cutoff + 1] = 0
    x = np.real(np.fft.ifft(X))

    return x


# Q1 and Q2
"""Plot the ECG"""
data = np.loadtxt('ECG_msc_matric_4.dat')
t_max = 20
t = np.linspace(0, t_max, len(data))

f0 = 50  # noise frequency
lR = 0.0001  # learning rate
fs = 250  # sample frequency
taps = (fs * 2)

"""Bandstop"""
f1 = (45 / fs)   # before 50Hz
f2 = (55 / fs)   # after 50Hz

"""Highpass"""
f3 = (0.5 / fs)   # ideal for cutting off DC noise

"""Call the function for Bandstop and Highpass"""
impulse_BS = bandstopDesign(f1, f2, taps)
impulse_HP = highpassDesign(f3, taps)

"""Convolve the coefficients of both the Bandstop and Highpass"""
coeff = convolveFull(impulse_HP, impulse_BS)

"""Reshuffle the coefficients"""
h1 = np.zeros(taps)
h1[0:int(taps / 2)] = coeff[int(taps / 2):taps]
h1[int(taps / 2):taps] = coeff[0:int(taps / 2)]
h_new = h1 * np.blackman(taps)

"""Call the class method dofilter, by passing in only a scalar value at a time which outputs a scalar value"""
fir = np.empty(max(len(data), len(h_new)) - min(len(data), len(h_new)) + 1)
fi = firfilter.firFilter(h_new)
for i in range(len(fir)):
    fir[i] = fi.dofilter(data[i])


# Q4

"""Find the range in the FIR plot where a heart beat occurs and plot it"""

plt.figure(2)
plt.subplot(2, 2, 1)
template = fir[525:725]
time = t[525:725]
plt.plot(time, template)
plt.title("matched filter template")
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

"""Plot the time reversed version if the template """
plt.subplot(2, 2, 2)
coefficients = template[::-1]
plt.plot(time, coefficients)
plt.title("matched filter time reversed")
plt.xlabel('time(sec)')
plt.ylabel('ECG (volts)')

"""Create the sinc function"""
# low = min(coefficients)
wavelet = np.linspace(-1, 1, len(time))
plt.subplot(2, 2, 3)
n_coeff = 5*np.sinc(wavelet*20)
n_coeff[0:int(len(time)/2)-10] = 0
n_coeff[int(len(time)/2) + 10:len(time)] = 0
plt.plot(time, n_coeff)
plt.title('Using sinc function')
# n_coeff = n_coeff**2

# """"Use the dofilter function with the highpass and the ecg data set to get a data set with no DC influence"""
# data1 = convolveValid(impulse_HP, data)
# data1 = np.empty(max(len(data), len(impulse_HP)) - min(len(data), len(impulse_HP)) + 1)
# fi = firfilter.firFilter(fs, impulse_HP)
# for i in range(len(data1)):
#     data1[i] = fi.dofilter(data[i])

""""Use the dofilter function with the wavelet and the previous data set to find the convolved data set"""
fir1 = np.empty(max(len(fir), len(n_coeff)) - min(len(fir), len(n_coeff)) + 1)
fi = firfilter.firFilter(n_coeff)
for i in range(len(fir1)):
    fir1[i] = fi.dofilter(fir[i])

# fir1 = np.empty(max(len(data1), len(n_coeff)) - min(len(data1), len(n_coeff)) + 1)
# fi = firfilter.firFilter(fs, n_coeff)
# for i in range(len(fir1)):
#     fir1[i] = fi.dofilter(data1[i])


"""Plot both the original FIR and the new FIR"""
t1 = np.linspace(0, t_max, len(fir))
t2 = np.linspace(0, t_max, len(fir1))
plt.figure(6)
plt.subplot(1, 2, 1)
plt.plot(t1, fir)
plt.title('Original FIR output')
plt.subplot(1, 2, 2)
plt.plot(t2, fir1)
# plt.xlim(2.5)
# plt.ylim(-.01,0.03)
plt.title('Sinc function on ECG with no DC noise')
plt.show()
