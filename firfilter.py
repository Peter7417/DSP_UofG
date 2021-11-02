import numpy as np


def getOutput(x, y):
    val = 0
    if len(x) == len(y):
        for i in range(len(x)):
            val += x[i] * y[i]
    else:
        raise 'cannot perform 1D array multiplication as array lengths are different'

    return val


class firFilter:

    def __init__(self, _data):
        self.coeff = _data
        self.ntaps = len(_data)
        self.buffer = np.zeros(self.ntaps)

        self.s_offset = 0

    def dofilter(self, v):
        # ring buffer
        self.buffer[self.s_offset % self.ntaps] = v
        
        output = 0
        for i in range(self.ntaps):
            output += self.buffer[(i + self.s_offset) % self.ntaps] * self.coeff[i]

        self.s_offset += 1
        return output

    def dofilterAdaptive(self, signal, noise, learningRate):

        for j in range(self.ntaps - 1):
            self.buffer[self.ntaps - j - 1] = self.buffer[self.ntaps - j - 2]
        self.buffer[0] = noise
        output = getOutput(self.buffer, self.coeff)
        error = signal - output
        for k in range(self.ntaps):
            self.coeff[k] += error * learningRate * self.buffer[k]

        return error
