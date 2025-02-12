import numpy as np
import matplotlib.pyplot as plt

def least_squares(x,y,N):#x:已知点，y:已知点函数值，N:多项式次数
    X = np.ones([len(x),N+1])
    for i in range(1,N+1):
        X[:,i] = x**i
    XTX = X.T@X
    XTX_inv = np.linalg.inv(XTX)
    b = XTX_inv@X.T@y
    return b,(np.sum((y-X@b)**2))

N1 = 4
x = np.arange(0,25,1)
y = np.array([15,14,14,14,14,15,16,18,20,22,23,25,28,29,31,30,28,27,25,24,22,20,18,17,16])
b,s = least_squares(x,y,N1)
print(b,s)

plt.scatter(x,y)
x1 = np.linspace(0,24,1201)
y1 = np.zeros(1201)
for i in range(N1+1):
    y1 += b[i]*(x1**i)
plt.plot(x1,y1)
plt.show()
