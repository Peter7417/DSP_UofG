import numpy as np

ls = []
r = np.loadtxt('ECG_msc_matric_5.dat')
print(r[0:1])
x = r[0:1]
y =[1, 1, 5, 5]
# b = np.zeros(len(x)+len(y)-1)
o = np.convolve(x, y)


l1 = len(x)
l2 = len(y)
print(l1)
N = l1 + l2 - 1
k = np.zeros(N)
# print(y[4])


for n in range(N):
    for i in range(l1):
        # print('k:', k)
        if 0 <= (n - i + 1) < l2:
            k[n] = k[n] + x[i] * y[n-i+1]


for i in k:
    ls.append((i))

print(ls)
ls.insert(0, ls.pop())
print(ls)
ls[0] = x[0]
print(ls)
print('convolve: ',o)
# print('function:',np.array(ls))
convolve = o
function = np.array(ls)

# print(convolve[20],function[20])
