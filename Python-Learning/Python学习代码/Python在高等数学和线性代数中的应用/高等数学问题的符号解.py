from mpmath import taylor
from sympy import *
import matplotlib.pyplot as plt
import numpy as np

"""--------------------求极限--------------------"""
print("--------------------求极限--------------------")
x=symbols('x')
lim1 = limit(sin(x)/x,x,0)
lim2 = limit((pow(1+1/x,x)),x,oo) #oo表示正无穷
print(lim1,lim2,sep=',')
print()

"""--------------------求导数--------------------"""
print("--------------------求导数--------------------")
x , y = symbols('x y')
z=sin(x)+x**2*exp(y)
print("关于x的二阶偏导为：",diff(z,x,2))
print("关于y的一阶偏导为：",diff(z,y))
print()

"""--------------------级数求和--------------------"""
print("--------------------级数求和--------------------")
k , n=symbols('k n')
sum1 = summation(k**2,(k,1,n))
sum1_factor = factor(sum1) #因式分解计算结果
sum2 = summation(1/(k**2),(k,1,oo)) #oo表示正无穷
print(sum1,sum1_factor,sum2,sep='\n')
print()

"""--------------------泰勒展开--------------------"""
print("--------------------泰勒展开--------------------")
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
x = symbols('x')
y = sin(x)

# 使用 lambdify 将符号函数转换为可用于数值计算的函数。
y_lambdify = lambdify(x,y,'numpy') #'numpy'表示将符号表达式转换为一个可以接受 NumPy 数组作为输入的函数

# 设置自变量范围
x_span = np.linspace(0,2,101)

# 逐级泰勒展开并绘图
for k in range(3,8,2):
    taylor = y.series(x,0,k) #等价于series(y,x,0,k)
    """.series(展开变量,展开位置,关于x的k次的高阶无穷小)"""
    taylor_removeO = taylor.removeO() #.removeO()表示去除余项
    taylor_lambdify = lambdify(x,taylor_removeO,'numpy')
    ax1.plot(x_span,taylor_lambdify(x_span))
    print(taylor) 
    print(taylor_removeO)
ax1.plot(x_span,y_lambdify(x_span))

# 将坐标轴移动到原点(0,0)
ax1.spines['bottom'].set_position('zero')
ax1.spines['left'].set_position('zero')

# 设置轴范围
ax1.set_xlim(0,2)
ax1.set_ylim(0,2)

# 隐藏上方和右侧的边框线
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

fig1.savefig("sin(x)泰勒展开.pdf",format='pdf')
print()

"""--------------------不定积分和定积分--------------------"""
print("--------------------不定积分和定积分--------------------")
x = symbols('x')

# 使用integrate()方法
integ1 = integrate(sin(2*x),(x,0,pi)) #求sin(2x)的定积分
integ1_indefinite = integrate(sin(x),x) #求sin(2x)的不定积分
integ2 = integrate(sin(x)/x,(x,0,oo)) #求sin(x)/x的定积分
integ2_indefinite = integrate(sin(x)/x,x) #求sin(x)/x的不定积分
print(integ1,integ1_indefinite,integ2,integ2_indefinite,sep='\n') #输出Si(x)表示正弦积分函数
print()

"""--------------------代数方程(方程组)的符号解--------------------"""
print("--------------------代数方程(方程组)的符号解--------------------")
x,y = symbols('x y')

# 使用solve()方法
# 解单变量方程
sol1 = solve(x**3-1,x) #解x^3=1
sol2 = solve((x-2)**2*(x-1)**3,x) #解(x-2)^2*(x-1)^3=0
print(sol1,sol2,sep='\n')
print(roots((x-2)**2*(x-1)**3)) #roots可以得到根的重数信息

# 解多变量方程组
sol3 = solve([x**2+y**2-1,x-y],[x,y]) #解方程组[x^2+y^2=1 , x-y=0]
print(sol3,sep='\n')
print()

"""--------------------微分方程(方程组)的符号解--------------------"""
print("--------------------代数方程(方程组)的符号解--------------------")
# 声明符号变量为函数类型
y = Function('y') #等价于y=symbols('y',cls=Function)
x = symbols('x')
eq1 = diff(y(x),x,2)-5*diff(y(x),x)+6*y(x) #齐次方程
eq2 = diff(y(x),x,2)-5*diff(y(x),x)+6*y(x)-x*exp(2*x) #非齐次方程
# 使用dsolve()方法
print("齐次方程的解为：",dsolve(eq1,y(x)))
print("非齐次方程的解为：",dsolve(eq1,y(x)))

# 初值问题：eq1,{y(0)=1,y'(0)=0}
print("初值问题的解为：",
      dsolve(eq1,y(x),
             ics={y(0):1,diff(y(x),x).subs(x,0):0}) #subs()方法用于替换表达式中的变量
      )
# 边值问题：eq2,{y(0)=1,y(2)=0}
print("边值问题的解为：",dsolve(eq2,y(x),ics={y(0):1,y(2):0}))

print()
