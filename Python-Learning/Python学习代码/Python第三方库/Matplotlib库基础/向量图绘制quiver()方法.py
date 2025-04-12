import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
axe = fig.add_subplot(111)
x = np.linspace(0, 15, 11)
y = np.linspace(0,10,11)
x, y = np.meshgrid(x, y)
v1 = y*np.cos(x)
v2 = y*np.sin(x)
axe.quiver(x, y, v1, v2)

plt.show()