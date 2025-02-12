import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
axe = fig.add_subplot(111, projection='3d')

t = np.linspace(0, 2*np.pi, 101)
s = np.linspace(-1,1,101)
t,s = np.meshgrid(t,s)

x = (2+s/2*np.cos(t/2))*np.cos(t)
y = (2+s/2*np.cos(t/2))*np.sin(t)
z = s/2*np.sin(t/2)

axe.plot_surface(x,y,z,cmap='viridis')
plt.axis('equal')
plt.show()