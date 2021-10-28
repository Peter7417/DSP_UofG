import numpy as np


class firFilter:

    def __init__(self, data):
        self.coeff = data
        self.ntaps = len(data)
        self.h = np.zeros(self.ntaps)
        self.ntaps = len(data)
        self.h1 = np.zeros(self.ntaps)

    def dofilter(self, v):
        # delay line buffer
        j = self.ntaps - 1
        while j > 0:
            self.h[j] = self.h[j - 1]
            j -= 1

        self.h[0] = v
        output = 0
        i = 0
        while i < self.ntaps:
            output += self.h[i] * self.coeff[i]
            i += 1

        return output

    def getOutput(self, x, y):
        val = 0
        if len(x) == len(y):
            for i in range(len(x)):
                val += x[i] * y[i]
        else:
            raise 'cannot perform 1D array multiplication as array lengths are different'

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
