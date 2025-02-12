import numpy
import numpy as np
def newton(u:float,X:numpy.ndarray,Y:numpy.ndarray,order:int)->float: #牛顿插值法，u:插值点，X:已知点，Y:已知点函数值，order:插值阶数
    n = len(X)
    if order >= n:
        print("阶数错误")
    else:
        table = np.zeros([n,n])
        table[:,0] = Y
        for k in range(1,n): #计算差商表
            for i in range(k,n):
                table[i,k] = (table[i,k-1]-table[i-1,k-1])/(X[i] - X[i-k])
        N = table[0,0] #初始化为常数项
        x_term = 1 #初始化含x的项为零次幂
        for power in range(1,order+1): #power表示现在进行到x的第几幂次
            x_term *= u-X[power-1]
            N += table[power,power]*x_term
        return N

x = np.array([0,0.3,0.6,0.9])
y = np.array([-1,-0.5950423577,0.09327128023,1.2136428])
is_run = True
while is_run:
    for u in np.linspace(0.3,0.6,300001):
        N = newton(u,x,y,3)
        if np.abs(N) < 0.000001:
            is_run = False
            break
        else:
            continue
print(u)

