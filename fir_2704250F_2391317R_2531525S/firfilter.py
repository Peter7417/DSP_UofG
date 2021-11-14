import numpy as np


class firFilter:

    def __init__(self, _data):
        self.coeff = _data
        self.ntaps = len(_data)
        self.buffer = np.zeros(self.ntaps)

        self.ringBuffIdx = 0  # BF 14/11/21 Renamed from offset -> ringBuffIdx

    def dofilter(self, v):
        # ring buffer
        # BF 14/11/21 removed modulo operation here since we now wrap back to
        # zero at the end of the function
        self.buffer[self.ringBuffIdx] = v

        output = 0
        for i in range(self.ntaps):
            output += self.buffer[(i + self.ringBuffIdx) % self.ntaps] * self.coeff[i]

        self.ringBuffIdx += 1

        #  BF 14/11/21 added this check to wrap the buffer index rather than
        #  using modulo
        if self.ringBuffIdx >= self.ntaps:
            self.ringBuffIdx = 0

        return output

    def dofilterAdaptive(self, signal, noise, learningRate):
        # ring buffer
        self.buffer[self.ringBuffIdx] = noise

        output = 0
        for i in range(self.ntaps):
            output += self.buffer[(i + self.ringBuffIdx) % self.ntaps] * self.coeff[i]

        # update coefficients
        error = signal - output
        for k in range(self.ntaps):
            self.coeff[k] += error * learningRate * self.buffer[(k + self.ringBuffIdx) % self.ntaps]

        self.ringBuffIdx += 1
        #  BF 14/11/21 added this check to wrap the buffer index rather than
        #  using modulo
        if self.ringBuffIdx >= self.ntaps:
            self.ringBuffIdx = 0

        return error
