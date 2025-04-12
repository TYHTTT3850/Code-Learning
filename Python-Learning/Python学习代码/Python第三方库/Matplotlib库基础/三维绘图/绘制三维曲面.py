import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.rc('text', usetex=True)

x = np.linspace(-6,6,30)
y = np.linspace(-6,6,30)
X,Y = np.meshgrid(x,y)
Z = np.sin(np.sqrt(X**2 + Y**2))
fig = plt.figure()

# 三维表面图形
axe1 = fig.add_subplot(121, projection='3d')
"""
add_subplot()参数解释
第一个数字(行数)：表示子图的总行数。
第二个数字(列数)：表示子图的总列数。
第三个数字(索引)：表示当前子图在整个布局中的索引位置，从左到右、从上到下依次编号。
"""
axe1.plot_surface(X,Y,Z,cmap='viridis')
axe1.set_xlabel('$X$')
axe1.set_ylabel('$Y$')
axe1.set_zlabel('$Z$')

# 三维网格图形
axe2 = fig.add_subplot(122, projection='3d')
axe2.plot_wireframe(X,Y,Z,color = 'c')
axe2.set_xlabel('$X$')
axe2.set_ylabel('$Y$')
axe2.set_zlabel('$Z$')

plt.show()