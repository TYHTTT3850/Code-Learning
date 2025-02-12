"""
GridSpec允许用户在一个图形中定义一个网格，并在这个网格中自由地放置多个子图(axes)。
相比于Figure对象的add_subplot()方法，Gridspec更加灵活。
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# 创建 GridSpec 对象，定义 3 行 3 列的网格
fig = plt.figure(figsize=(10, 6)) #创建一个Figure对象
gs = gridspec.GridSpec(3, 3) #创建一个GridSpec对象

# 子图1：占据整个左列 (0:3 行，0 列)
axe1 = fig.add_subplot(gs[0:3, 0])
axe1.plot(np.random.rand(10),'r-',label='axe1')
axe1.set_title('子图1：占据左列',fontproperties='SimSun')
axe1.legend(loc='best') #位置选择

# 子图2：占据上面两行的中间和右侧 (0:2 行，1:3 列)
axe2 = fig.add_subplot(gs[0:2, 1:3])
axe2.plot(np.random.rand(10))
axe2.set_title('子图2：占据上面两行的右侧',fontproperties='SimSun')

# 子图3：占据底部一行的中间 (2 行，1 列)
axe3 = fig.add_subplot(gs[2, 1])
axe3.plot(np.random.rand(10))
axe3.set_title('子图3：占据底部一行的中间',fontproperties='SimSun')

# 子图4：占据底部一行的右侧 (2 行，2 列)
axe4 = fig.add_subplot(gs[2, 2])
axe4.plot(np.random.rand(10))
axe4.set_title('子图4：占据底部一行的右侧',fontproperties='SimSun')

# 调整布局
plt.tight_layout() #自动调整子图布局的一个方法
plt.show()
