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
