import matplotlib.pyplot as plt
import numpy as np

# Figure 常用设置示例
fig = plt.figure(figsize=(10, 6))  #创建指定大小的图形
axe = fig.add_subplot(111) #绘制图形
x = np.linspace(0,2*np.pi,100)
axe.plot(x,np.sin(x))

# 基本设置
fig.suptitle('Global Title', fontsize=16, fontweight='bold')  #设置总标题

# 布局设置
fig.subplots_adjust(
    top=0.9,      #上边距
    bottom=0.1,   #下边距
    left=0.1,     #左边距
    right=0.9,    #右边距
    hspace=0.3,   #子图垂直间距
    wspace=0.3    #子图水平间距
)
fig.tight_layout() #自动调整布局

# 图片存储
fig.savefig('Figure对象常用设置.pdf', format='pdf',
    bbox_inches='tight',  #自动调整边距
    transparent=True   #透明背景
)
