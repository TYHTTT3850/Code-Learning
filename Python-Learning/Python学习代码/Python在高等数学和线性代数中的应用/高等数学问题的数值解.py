import numpy as np
import matplotlib.pyplot as plt
"""--------------------泰勒级数--------------------"""
# sinx泰勒级数
def fac(n): #阶乘
    if n < 1:
        return 1
    else:
        return n*fac(n-1)
def item(n,x): #每一项
    return (-1)**n*x**(2*n+1)/fac(2*n+1)
def mysin(n,x): #每一项求和
    if n < 0:
        return 0
    else:
        return mysin(n-1,x)+item(n,x)
x = np.linspace(-2*np.pi,2*np.pi,101)
fig1 = plt.figure()
axe1 = fig1.add_subplot(111)
axe1.plot(x,np.sin(x),label='sin')
for n in range(1,6,2):
    axe1.plot(x,mysin(n,x),label=n)
axe1.legend(loc='best')

"""--------------------数值微分--------------------"""
d = 200.0
xy = np.array([[-d,d],[d,d],[d,-d],[-d,-d]]) #初始位置
T = np.linspace(0,400,201) #划分时刻
dt = 2
v = 1.0
xyn = np.empty((4,2)) #下一个时刻的位置数组
Txy = xy #所有时刻的位置数组
for t in T:
    for i in [0,1,2,3]:
        j = (i + 1) % 4
        dxy = xy[j] - xy[i]  # 每个人的朝向
        dd = dxy / np.linalg.norm(dxy)  # 单位化向量
        xyn[i] = xy[i]+v*dt*dd #计算下一步的位置
    Txy = np.hstack((Txy,xyn))
    xy = xyn
fig2 = plt.figure()
axe2 = fig2.add_subplot(111)
name = ['甲','乙','丙','丁']
for i in [0,1,2,3]:
    axe2.plot(Txy[i,0::2],Txy[i,1::2])
axe2.set_title("四人相对行走",font='SimSun')

"""--------------------数值积分--------------------"""

# 一重积分
from scipy.integrate import quad
f = lambda x: np.sin(np.sqrt(np.cos(x)+x**2))
print(quad(f,0,1))

# 多重积分
"""多重积分要化为累次积分"""

# 二重积分
from scipy.integrate import dblquad
f1 = lambda y,x: x*y**2 #被积函数格式为f(y,x)，表示先对y积分再对x积分，必须严格按照积分顺序书写自变量顺序
f2 = lambda y,x: np.exp(-x**2/2)*np.sin(x**2+y)
print("I1=",dblquad(f1,a=0,b=2,gfun=0,hfun=1))
print("I2=",dblquad(f2,a=-1,b=1,gfun=lambda x:-np.sqrt(1-x**2),hfun=lambda x:np.sqrt(1-x**2)))
"""积分区间：x从 a 到 b，y 从 gfun(x) 到 hfun(x) """

# 三重积分
# #被积函数格式为f(z,y,x)，表示先对z积分再对y积分最后对x积分，必须严格按照积分顺序书写自变量顺序
from scipy.integrate import tplquad
f3 = lambda z,y,x: z*np.sqrt(x**2+y**2+1) #积分区域为x^2+y^-2x=0与z=0和z=6围成的区域
print("I3=",tplquad(f3,a=0,b=2,gfun=lambda x:-np.sqrt(2*x-x**2),hfun=lambda x:np.sqrt(2*x-x**2),
                    qfun=0,rfun=6))
"""积分区间：x从 a 到 b，y 从 gfun(x) 到 hfun(x)，z从qfun(x,y) 到 rfun(x,y)"""

"""--------------------非线性方程(组)数值解--------------------"""
from scipy.optimize import fsolve

# 求解x^3 + 1.1*x^2 + 0.9*x - 1.4 = 0
eq1 = lambda x: x**3 + 1.1*x**2 + 0.9*x - 1.4
print("Sol1:",fsolve(eq1,0))

# 求解方程组
def eq2(x:list):
    return [5*x[1]+3,4*x[0]**2-2*np.sin(x[1]*x[2]),x[1]*x[2]-1.5]
print("Sol2:",fsolve(eq2,[1,1,1]))

"""--------------------函数极值点的数值解--------------------"""

# 求函数e^x*cos(2x)在[0,3]上的极小点
from scipy.optimize import fminbound
f = lambda x: np.exp(x)*np.cos(2*x)
x0 = fminbound(f,0,3)
print(f"极小点为{x0},极小值为{f(x0)}")

# 求函数e^x*cos(2x)在0附近的极小点
from scipy.optimize import fmin
f = lambda x: np.exp(x)*np.cos(2*x)
x0 = fmin(f,0)
print(f"极小点为{x0},极小值为{f(x0)}")

# 多元函数的极值点
from scipy.optimize import minimize #基本调用格式 minimize(fun, x0, args=(), method=None),method表示使用的算法
f = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
x0 = minimize(f,[2,2])
print(f"极小点为{x0.x},极小值为{x0.fun}")
