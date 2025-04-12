import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rc('text', usetex=True)

fig = plt.figure()

# 绘制等高线
axe1 = fig.add_subplot(121)
z = np.loadtxt('高程数据表.txt')
x = np.arange(0,1500,100)
y = np.arange(1200,-10,-100)
contr = axe1.contour(x,y,z) #绘制等高线
axe1.clabel(contr,colors="black") #添加等高线上的标签并将字色设置为黑色
axe1.set_xlabel('$x$')
axe1.set_ylabel('$y$',rotation=0)

# 绘制三维表面图
X,Y = np.meshgrid(x,y)
axe2 = fig.add_subplot(122,projection='3d')
axe2.plot_surface(X,Y,z,cmap="viridis")
axe2.set_xlabel('$x$')
axe2.set_ylabel('$y$')
axe2.set_zlabel('$z$')

plt.show()