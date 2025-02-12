import numpy as np
def newton(u,X,Y,order): #牛顿插值法，u:插值点，X:已知点，Y:已知点函数值，order:插值阶数
    n = len(X)
    if order >= n:
        print("阶数错误")
    else:
        table = np.zeros([n,n])
        table[:,0] = Y
        for k in range(1,n):
            for i in range(k,n):
                table[i,k] = (table[i,k-1]-table[i-1,k-1])/(X[i] - X[i-k])
        N = table[0,0] #初始化为常数项
        x_term = 1 #初始化含x的项为零次幂
        for power in range(1,order+1): #power表示现在插值到x的第几次幂
            x_term *= u-X[power-1]
            N += table[power,power]*x_term
        print(table)
        print(N)

x = np.array([0.5,0.6,0.4])
y = np.array([-0.6931,-0.5108,-0.9163,])
newton(0.54,x,y,2)
