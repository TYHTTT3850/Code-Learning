import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)

def runge_kutta(f, x0, y0, h, x_end):
    """
        dy/dx = f(x, y)
        参数:
            f: 函数 f(x, y)，表示 dy/dx
            x0: 初值 x
            y0: 初值 y
            h: 步长
            x_end: 终点
            n: 使用的 R-K 法的阶数

        返回:
            x: 离散化的 x 值
            y: 离散化的 y 值
        """
    # 初始化
    x = [x0]
    y = [y0]
    while x[-1] < x_end:
        k1 = f(x[-1], y[-1])
        k2 = f(x[-1] + h/2, y[-1] + h/2 * k1)
        k3 = f(x[-1] + h/2, y[-1] + h/2 * k2)
        k4 = f(x[-1] + h, y[-1] + h*k3)
        x.append(x[-1]+h)
        y.append(y[-1] + h/6 * (k1 + 2*k2 + 2*k3 + k4))
    return np.array(x), np.array(y)

def f(x, y):
    return y/x * (np.log(y) - np.log(x))

X , numerical_solution = runge_kutta(f, 1, 1, 0.01, 2)

# 绘图
fig = plt.figure()

# 数值解
axe1 = fig.add_subplot(121)
axe1.plot(X, numerical_solution,'g-')
axe1.set_xlabel('$x$')
axe1.set_ylabel('$y$',rotation=0)
axe1.set_title("numerical solution")

# 解析解
axe2 = fig.add_subplot(122)
axe2.plot(X, X*(np.exp(1-X)),'r-')
axe2.set_xlabel('$x$')
axe2.set_ylabel('$y$',rotation=0)
axe2.set_title("analytical solution")

fig.tight_layout()
plt.savefig('Runge_Kutta方法.pdf', format='pdf')