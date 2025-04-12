"""
Axes.spines详解(3D图形没有spines)
spines指的是围绕绘图区域的边界线,也就是图表的"边框 "。一个典型的图表有四条边框线:
'top' - 顶部边框
'bottom' - 底部边框(通常是 x 轴所在位置)
'left' - 左侧边框(通常是 y 轴所在位置)
'right' - 右侧边框
"""

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)  # 添加一个子图
x = np.linspace(0, 2*np.pi, 100)
ax.plot(x, np.sin(x),label='sin(x)')

"""--------------------子图图例设置--------------------"""
ax.legend(loc='best') #自动选择最好的位置

"""--------------------子图标题和轴标签设置--------------------"""
ax.set_title('Subplot Title')
ax.set_xlabel('X',loc='right') #标签位置设置为最右边
ax.set_ylabel('Y',rotation=0,loc='top') #标签位置设置为最上边

"""--------------------子图轴范围设置--------------------"""
ax.set_xlim(0,2*np.pi)
ax.set_ylim(-1, 1)

"""--------------------子图边框线设置--------------------"""
# 将坐标轴移动到原点(0,0)
ax.spines['bottom'].set_position('zero') #bottom <——> x 轴
ax.spines['left'].set_position('zero') #left <——> y 轴

# 隐藏上方和右侧的边框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 为左侧和底部边框设置不同的样式
ax.spines['left'].set_linestyle('-.')    #点划线
ax.spines['bottom'].set_linestyle(':')   #虚线

# 设置边框线的颜色
ax.spines['left'].set_color('green')
ax.spines['bottom'].set_color('blue')

"""--------------------子图网格线设置--------------------"""
ax.grid(True, linestyle='--', alpha=0.7)

"""--------------------子图背景色设置--------------------"""
ax.set_facecolor('#f0f0f0')

plt.show()
