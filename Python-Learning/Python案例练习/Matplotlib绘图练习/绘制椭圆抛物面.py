import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
axe = fig.add_subplot(111, projection='3d')

# 参数准备
x = np.linspace(-5,5,1001)
y = np.linspace(-5,5,1001)
x,y = np.meshgrid(x,y)

z = x**2/10 + y**2/6

axe.plot_surface(x,y,z,cmap='viridis')
axe.set_xlabel('X')
axe.set_ylabel('Y')
axe.set_zlabel('Z')

plt.axis('equal')
plt.show()