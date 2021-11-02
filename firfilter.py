import numpy as np


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
        # ring buffer
        self.buffer[self.s_offset % self.ntaps] = noise
        
        output = 0
        for i in range(self.ntaps):
            output += self.buffer[(i + self.s_offset) % self.ntaps] * self.coeff[i]

        # update coefficients
        error = signal - output
        for k in range(self.ntaps):
            self.coeff[k] += error * learningRate * self.buffer[(k + self.s_offset) % self.ntaps]

        self.s_offset += 1
        return error