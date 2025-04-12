"""
子图位置与编号：从上到下，从左到右，从1开始依次编号，子图之间可以合并
以2×3布局为例，子图位置与编号分别为：
1  ,  2  ,  3
4  ,  5  ,  6
若子图之间相互合并，如:
1  ,  2  ,  (3,6)
(4,5合并) ,  合并
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pgf') #修改绘图后端以确保中文支持，但pgf不支持交互式绘图，所以只能通过保存的方式绘制。
plt.rc('text', usetex=True) #启用tex渲染

"""--------------------绘制2×2的规则子图--------------------"""
fig = plt.figure(figsize=(8,6))
x = np.linspace(0, 2*np.pi, 200)

# 1号子图
axe1 = fig.add_subplot(2,2,1)
axe1.plot(x, np.sin(x))
axe1.set_xlabel('$x$')
axe1.set_ylabel('$y$',rotation=0)

# 2号子图
axe2 = fig.add_subplot(2,2,2)
axe2.plot(x, np.cos(x),linestyle='--')
axe2.set_xlabel('$x$')
axe2.set_ylabel('$y$',rotation=0)

# 3号子图
axe3 = fig.add_subplot(2,2,3)
axe3.plot(x, x*np.sin(x),linestyle='--')
axe3.set_xlabel('$x$')
axe3.set_ylabel('$y$',rotation=0)

# 4号子图
axe4 = fig.add_subplot(2,2,4)
axe4.plot(x, np.sin(x**2),linestyle='--')
axe4.set_xlabel('$x$')
axe4.set_ylabel('$y$',rotation=0)

fig.suptitle("2×2规则子图",fontproperties='SimSun')
fig.tight_layout() #自动调整子图间布局
fig.savefig('2×2规则子图.pdf',format='pdf')

"""--------------------绘制2×3的不规则子图--------------------"""
fig1 = plt.figure(figsize=(8,6))

# 1号子图
axe1_1 = fig1.add_subplot(2,3,1)
axe1_1.plot(x, np.sin(x))
axe1_1.set_xlabel('$x$')
axe1_1.set_ylabel('$y$',rotation=0)

# 2号子图
axe2_2 = fig1.add_subplot(2,3,2)
axe2_2.plot(x, np.cos(x),linestyle='--')
axe2_2.set_xlabel('$x$')
axe2_2.set_ylabel('$y$',rotation=0)

# (3,6)合并子图
axe3_6 = fig1.add_subplot(2,3,(3,6))
axe3_6.plot(x, np.sin(x**2),linestyle='--')
axe3_6.set_xlabel('$x$')
axe3_6.set_ylabel('$y$',rotation=0)

# (4,5)合并子图
axe4_5 = fig1.add_subplot(2,3,(4,5))
axe4_5.plot(x, x*np.sin(x),linestyle='--')
axe4_5.set_xlabel('$x$')
axe4_5.set_ylabel('$y$',rotation=0)

fig1.suptitle("2×3不规则子图",fontproperties='SimSun')
fig1.tight_layout()
fig1.savefig('2×3不规则子图.pdf',format='pdf')
