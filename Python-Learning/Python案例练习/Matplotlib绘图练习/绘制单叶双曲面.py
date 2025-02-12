import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 创建图形和3D坐标轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 生成连续的参数范围
t = np.linspace(-2, 2, 101)
u = np.linspace(0, 2 * np.pi, 101)
t, u = np.meshgrid(t, u)

# 参数方程参数
a = np.sqrt(8)
b = np.sqrt(10)
c = np.sqrt(6)
# 计算坐标
x = a * np.cosh(t) * np.cos(u)
y = b * np.cosh(t) * np.sin(u)
z = c * np.sinh(t)

# 绘制曲面
surface = ax.plot_surface(x, y, z,
                            cmap='viridis',alpha=0.7,
                            edgecolor='None')



# 绘制Z=0处椭圆参考线
ax.plot3D(a*np.cos(u),b*np.sin(u),np.zeros_like(u),color='blue')
ax.scatter(0,b,0,color='red')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置标题
ax.set_title('连续的单叶双曲面',fontproperties='SimSun')

plt.axis('equal')
plt.show()
