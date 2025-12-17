# To implement following activation function (i) purelin(n),(ii) binary threshold :hardlim(n),(iii)haradlims(n),(iv)Tansig(n), (v)logsig(n).
import numpy as np

def purelin(n):
    return n

def hardlim(n):
    return np.where(n >= 0,1,0)

def hardlims(n):
    return np.where(n <= 0,1,-1)

def tansig(n):
    return np.tanh(n)

def logsig(n):
    return 1/(1+np.exp(-n))

x = np.array([-2,-1,0,1,2])

print("purelin:",purelin(x))
print("hardlim:",hardlim(x))
print("hardlims:",hardlims(x))
print("tansig:",tansig(x))
print("logsig:",logsig(x))