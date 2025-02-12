import numpy as np

def Trapezoidal(f,a,b):
    I = (b-a) * (f(a) + f(b)) / 2
    return I

def Simpson(f,a,b):
    I = (b-a) * (f(a) + 4*f(a/2 + b/2) + f(b)) / 6
    return I

def Cotes(f,a,b):
    I = (b-a) * (7*f(a) + 32*f(3*a/4 + b/4) + 12*f(a/2 + b/2) + 32*f(a/4 + 3*b/4) +7*f(b)) / 90
    return I

def Numeric_integral(f,a,b,eps,method):#f:积分函数，a:积分下限，b:积分上限，eps:精度控制 method:求积方法
    #初始化I1和I2
    I1 = method(f,a,b)
    I2 = 0

    #从分为2段区间的情况开始
    for i in range(1, int(1 / eps)+1):
        n = 2 ** i  #表示分段区间数
        h = (b - a) / n  #表示分段区间间隔

        # 分段求积
        for t2 in range(0, n): #遍历所有分段区间
            I2 += method(f,a + t2*h,a + t2*h + h)

        # 判断结果是否符合要求
        if np.abs(I2 - I1) < eps:
            return n, I2
        elif i == int(1 / eps): #在最大分段区间数的情况下仍无满足要求的近似值，则返回此时的近似值
            return n, I2
        else:
            I1 = I2 #I2的值保存至I1
            I2 = 0 #重置I2的值

# 定义求积函数
def f(x):
    if x == 0:
        return 1
    else:
        return np.sin(x)/x

eps = 0.00001
n, I = Numeric_integral(f,0,1,eps,method=Simpson)
print(n,I)