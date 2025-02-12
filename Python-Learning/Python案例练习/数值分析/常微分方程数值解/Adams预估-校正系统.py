import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

def adams_predictor_corrector(f, x0, y0, h, x_end):
    """
    参数：
        f: 函数 dy/dx = f(x, y)
        x0: 初始自变量值
        y0: 初始因变量值
        x_end: 结束自变量值
        h: 步长
    返回值：
        自变量数组和对应的数值解数组
    """
    # 初始化
    x = [x0]
    y = [y0]

    # 使用四阶Runge-Kutta方法计算前四个初值
    def runge_kutta_4(f, x, y, h):
        k1 = f(x, y)
        k2 = f(x + h / 2, y + h * k1 / 2)
        k3 = f(x + h / 2, y + h * k2 / 2)
        k4 = f(x + h, y + h * k3)
        return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    for i in range(1,4):  # 计算f1,f2,f3
        x_next = x[-1] + h
        y_next = runge_kutta_4(f, x[-1], y[-1], h)
        x.append(x_next)
        y.append(y_next)

    # 进入Adams预估-校正
    while x[-1] < x_end:

        # 预测
        y_predict = y[-1] + h/24 * (55 * f(x[-1], y[-1]) - 59 * f(x[-2], y[-2]) + 37 * f(x[-3], y[-3]) - 9 * f(x[-4], y[-4]))

        # Adams-Moulton校正
        y_correct = y[-1] + h/24 * (9 * f(x[-1] + h, y_predict) + 19 * f(x[-1], y[-1]) - 5 * f(x[-2], y[-2]) + f(x[-3], y[-3]))

        # 更新结果
        x.append(x[-1] + h)
        y.append(y_correct)

    return np.array(x), np.array(y)

def f(x, y):
    return y/x * (np.log(y) - np.log(x))

X , numerical_solution = adams_predictor_corrector(f, 1, 1, 0.01, 2)

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
plt.savefig('Adams预估-校正.pdf', format='pdf')