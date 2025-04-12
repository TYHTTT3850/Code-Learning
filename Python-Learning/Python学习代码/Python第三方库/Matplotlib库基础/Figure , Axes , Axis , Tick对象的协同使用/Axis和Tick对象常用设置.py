import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)  # 添加一个子图
x = np.linspace(0, 2*np.pi, 100)
ax.plot(x, np.sin(x),label='sin(x)')

"""--------------------主次刻度--------------------"""
# 设置主刻度长度和粗细
ax.tick_params(axis='both', which='major', length=8, width=1.5) #axis='both'表示对x和y轴都生效

# 设置次刻度长度和粗细
ax.tick_params(axis='both', which='minor', length=4, width=1,color='red')

# 添加并设置x轴主刻度和次刻度之间的间距
ax.xaxis.set_major_locator(plt.MultipleLocator(2))   #每2个单位长度一个主刻度
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5)) #每0.5个单位长度次刻度

# 添加并设置y轴主刻度和次刻度之间的间距
ax.yaxis.set_major_locator(plt.MultipleLocator(0.25)) #每0.25个单位长度一个主刻度
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.125)) #每0.125个单位长度一个次刻度 

"""--------------------x,y坐标轴--------------------"""
# x轴设置
ax.xaxis.set_label_text('X')
ax.xaxis.label.set_fontsize(8)
ax.xaxis.set_ticks_position('both') #顶部和底部都显示X轴刻度，不设置的话仅底部有

# y轴设置
ax.yaxis.set_label_text('Y',rotation=0)
ax.yaxis.label.set_fontsize(8)
ax.yaxis.set_ticks_position('both') #左侧和右侧都显示Y轴刻度，不设置的话仅左部有

plt.show()
