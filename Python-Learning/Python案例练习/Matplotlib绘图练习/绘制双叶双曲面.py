import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
axe = fig.add_subplot(111, projection='3d')

# 参数准备
u = np.linspace(-2,2,101)
t = np.linspace(0,2*np.pi,101)
u,t = np.meshgrid(u,t)

# 绘制曲面
y = np.sqrt(12)*np.sinh(u)*np.cos(t)
z = np.sqrt(8)*np.sinh(u)*np.sin(t)
x_pos = np.cosh(u)
x_neg = -np.cosh(u)
axe.plot_surface(x_pos,y,z,cmap='viridis',alpha=0.7)
axe.plot_surface(x_neg,y,z,cmap='viridis',alpha=0.7)

# 绘制X=0处的参考平面
y_r = 5*u
z_r = 10*np.sin(t)
axe.plot_surface(np.zeros_like(u),y_r,z_r,
                 color='grey',alpha=0.1)

# 图片设置
axe.set_xlabel('X')
axe.set_ylabel('Y')
axe.set_zlabel('Z')

plt.axis('equal')
plt.show()