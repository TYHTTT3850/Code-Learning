import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 时间和销售数据
t0 = np.arange(1,7)
x0 = np.array([5.081,4.611,5.1177,9.3775,11.0574,11.0524])


# 最小二乘法线性拟合
l1 = np.polyfit(t0,x0,1)
print(f"线性拟合结果：y = {l1[0]}x + {l1[1]}")
xh1 = np.polyval(l1,t0) #计算预测值
delta1 = np.abs(x0-xh1)/x0 #计算相对误差

# 指数拟合
x1 = np.cumsum(x0) #一次累加

def function(t,a,b,c): #定义拟合曲线
    return a*np.exp(b*t) + c
para,cov = curve_fit(function,t0,x1) #拟合

xh21 = function(t0,*para) #计算累加数列拟合值
xh22 = np.diff(xh21) #计算数组相邻元素之间的差,x0(k) = x1(k) - x1(k-1)
xh22 = np.hstack([xh21[0],xh22]) #x0(1) = x1(1)
delta2 = np.abs(x0-xh22)/x0

fig = plt.figure()
axe1 = fig.add_subplot(131)
axe2 = fig.add_subplot(132)
axe3 = fig.add_subplot(133)

axe1.scatter(t0,x0)
axe1.plot(t0,xh1,'r--')
axe1.set_title("线性拟合",font="SimSun")

axe2.scatter(t0,x1)
axe2.plot(t0,xh21,'r--')
axe2.set_title("累加后拟合",font="SimSun")

axe3.scatter(t0,x0)
axe3.plot(t0,xh22,'r--')
axe3.set_title("累减还原检验",font="SimSun")
plt.show()