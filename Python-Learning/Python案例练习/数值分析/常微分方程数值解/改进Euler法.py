import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

def improved_euler(f, x0, y0, h, x_end):
    """
    dy/dx = f(x, y)
    参数:
        f: 函数 f(x, y)，表示 dy/dx
        x0: 初值 x
        y0: 初值 y
        h: 步长
        x_end: 终点

    返回:
        np.array(x): 离散化的 x 值
        np.array(y): 离散化的 y 值
    """

    #初始化x和y
    x = [x0]
    y = [y0]
    while x[-1] < x_end:
        xk = x[-1] + h
        yk_predict = y[-1]+h * f(x[-1], y[-1])
        yk = y[-1] + h/2 * (f(x[-1],y[-1]) + f(xk, yk_predict))
        x.append(xk)
        y.append(yk)
    return np.array(x), np.array(y)

def f(x,y):
    return y/x + x * np.exp(x)

X, numerical_solution = improved_euler(f,1,0,0.01,2)

# 绘图
fig = plt.figure()
axe1 = fig.add_subplot(121)
axe2 = fig.add_subplot(122)

# 数值解
axe1.plot(X, numerical_solution,'g-')
axe1.set_xlabel('$x$')
axe1.set_ylabel('$y$',rotation=0)
axe1.set_title("numerical solution")

# 解析解
axe2.plot(X, X*(np.exp(X)-np.e),'r-')
axe2.set_xlabel('$x$')
axe2.set_ylabel('$y$',rotation=0)
axe2.set_title("analytical solution")

fig.tight_layout()
plt.savefig('改进Euler法.pdf',format='pdf')