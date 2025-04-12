import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

fig = plt.figure()
axe = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d') #设置三维图形模式

z = np.linspace(0, 100, 1000)
x = np.sin(z)*z  # x 轴数据
y = np.cos(z)*z  # y 轴数据
axe.plot3D(x, y, z) #plot3D()等同于plot()，只是为了强调该方法是用于绘制三维数据。
plt.show()
