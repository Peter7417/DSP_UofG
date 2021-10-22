import numpy as np


class firFilter:

    def __init__(self, fs, data):
        self.fs = fs
        self.data = data

    def dofilter(self, v):
        ls = []
        h = np.zeros(self.fs * 2)
        h[0:int(self.fs * 2 / 2)] = v[int(self.fs * 2 / 2):self.fs * 2]
        h[int(self.fs * 2 / 2):self.fs * 2] = v[0:int(self.fs * 2 / 2)]

        h_new = h * np.blackman(self.fs * 2)

        l1 = len(self.data)
        l2 = len(h_new)

        N = l1 + l2 - 1
        k = np.zeros(N)

        for n in range(N):
            for i in range(l1):
                if 0 <= (n - i + 1) < l2:
                    k[n] = k[n] + self.data[i] * h_new[n - i + 1]

        for i in k:
            ls.append(i)

        ls.insert(0, ls.pop())
        ls[0] = self.data[0]
        conv = np.array(ls)

        # conv = np.convolve(h_new, self.data, mode='valid')
        return conv
