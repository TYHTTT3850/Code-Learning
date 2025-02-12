import matplotlib.pyplot as plt
import numpy as np

"""--------------------创建示例数据--------------------"""
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

"""--------------------创建Figure对象和子图--------------------"""
fig = plt.figure()
ax1 = fig.add_subplot(211)  #上半部分子图
ax2 = fig.add_subplot(212)  #下半部分子图

# 在两个子图中绘制数据
line1= ax1.plot(x, y1, 'b-', label='sin(x)')
line2= ax2.plot(x, y2, 'r-', label='cos(x)')

# 设置Figure标题
fig.suptitle('Matplotlib对象综合应用示例', fontproperties='SimSun')

"""--------------------自定义Axes对象--------------------"""
for ax in [ax1, ax2]:
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.7)

    # 设置背景色
    ax.set_facecolor('#000000') #纯黑背景色

    # 设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('red') #设置上下左右四个边框线的颜色

# 设置每个子图的具体属性
ax1.set_title('Sin',fontsize=10)
ax1.set_xlim(0, 10)
ax1.set_ylim(-1.2, 1.2)
ax1.legend(loc='upper right')
ax2.set_title('Cos',fontsize=10)
ax2.set_xlim(0, 10)
ax2.set_ylim(-1.2, 1.2)
ax2.legend(loc='upper right')

"""--------------------自定义Axis对象(坐标轴)--------------------"""
for ax in [ax1, ax2]:
    # X轴设置
    ax.xaxis.set_label_text('X')
    ax.xaxis.label.set_fontsize(8)
    ax.xaxis.set_ticks_position('both') #顶部和底部都显示X轴

    # Y轴设置
    ax.yaxis.set_label_text('Y',rotation=0)
    ax.yaxis.label.set_fontsize(8)
    ax.yaxis.set_ticks_position('both') #左侧和右侧都显示Y轴


"""--------------------自定义Tick对象(刻度)--------------------"""
for ax in [ax1, ax2]:
    # 主刻度设置
    ax.tick_params(axis='both', which='major', length=8, width=1.5)

    # 次刻度设置
    ax.tick_params(axis='both', which='minor', length=4, width=1)

    # 添加次刻度
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))


"""--------------------调整子图间距--------------------"""
plt.tight_layout()
plt.subplots_adjust(top=0.9)  #为总标题留出空间

# 显示图形
plt.show()