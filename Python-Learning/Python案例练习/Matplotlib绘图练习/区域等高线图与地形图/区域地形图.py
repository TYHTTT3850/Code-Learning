import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.rc('text', usetex=True)

data = pd.read_excel("区域高程数据.xlsx",header=None)
fig = plt.figure()
axe = fig.add_subplot(111, projection='3d')

z = np.array(data.head(874),dtype=int)/1000
m,n = z.shape
x = 50*np.arange(0,n)/1000
y = 50*np.arange(0,m)/1000
x,y = np.meshgrid(x,y)
Topography = axe.plot_surface(x,y,z,cmap="viridis")
cbar = plt.colorbar(Topography) #创建颜色条
cbar.set_label("Height(KM)") #设置颜色条标签
axe.set_xlabel('$x$')
axe.set_ylabel('$y$')
axe.set_zlabel('$z$')
axe.set_title("The Topography of This Area")
fig.tight_layout()
plt.show()