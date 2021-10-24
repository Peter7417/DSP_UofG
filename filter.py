import numpy as np


class firFilter:

    def __init__(self, fs, data):
        self.fs = fs
        self.data = data
        self.taps = fs * 2
        self.h = np.zeros(self.taps)
        self.ntaps = len(data)
        self.h1 = np.zeros(self.ntaps)
        self.coeff = self.data

    def dofilter(self, v):
        ls = []
        for i in self.data:
            ls.append(i)

        if 0 <= ls.index(v) <= int(self.taps / 2) - 1:
            distance = ls.index(v) - 0
            self.h[int(self.taps / 2) + distance - 1] = v

        if int(self.taps / 2) <= ls.index(v) <= self.taps - 1:
            distance = ls.index(v) - int(self.taps / 2)
            self.h[0 + distance - 1] = v

        h_new = self.h * np.blackman(self.taps)
        return h_new

    def getOutput(self, x, y):
        val = 0
        if len(x) == len(y):
            for i in range(len(x)):
                val += x[i] * y[i]
        else:
            raise 'cannot perform 1D array multiplication as array lengths are not the same'

        return val

    def dofilterAdaptive(self, signal, noise, learningRate):

        for j in range(self.ntaps - 1):
            self.h1[self.ntaps - j - 1] = self.h1[self.ntaps - j - 2]
        self.h1[0] = noise
        output = self.getOutput(self.h1, self.coeff)
        error = signal - output
        for k in range(self.ntaps):
            self.coeff[k] += error * learningRate * self.h1[k]

        return error
