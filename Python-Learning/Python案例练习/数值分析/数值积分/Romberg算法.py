import numpy as np

def Romberg(f,a,b,eps):
    M = 11 #最大对分区间数2^(11-1)
    T = np.zeros([M,M])
    T[0,0] = (b-a) * (f(a) + f(b)) / 2 #取n=1,n=2^k,即k=0

    # 从对分两段区间开始
    for k in range(1, M): #n=2^k,2n=2^(k+1)
        n = 2**k

        # 梯形积分递推公式
        T[k,0] = 1/2 * T[k-1,0]
        for j in range(1, n, 2):
            T[k,0] += (b-a)/n * f(a + j*(b-a)/n)
        
        # 外推加速公式
        for i in range(1, k+1):
            T[k,i] = (4**i * T[k,i-1] - T[k-1,i-1]) / (4**i - 1)

        if np.abs(T[k,k]-T[k-1,k-1]) < eps: #满足精度
            return k, T[k,k]

        elif k == M-1: #达到最大对分区间数
            return k, T[k,k]

def f(x):
    if x == 0:
        return 1
    else:
        return np.sin(x)/x
eps = 0.0000001

k , I = Romberg(f,0,1,eps)
print(k , I)
