"""
Matplotlib绘图主要步骤：
1、导入Matplotlib.pyplot模块。
2、设置绘图的数据和参数。
3、调用 Matplotlib.pyplot模块的plot()、pie()、bar()、hist()、scatter()等方法。
4、设置绘图的 x轴、y轴、标题、网格线、图例等内容。
5、调用show()方法显示已绘制的图形。
"""

import numpy as np,pandas as pd
import matplotlib.pyplot as plt
x = np.linspace(0, 10, 1000)

"""--------------------一个图中单条曲线--------------------"""
plt.plot(x, np.sin(x)) #画sin(x)图像
plt.show() #展示图片

"""--------------------同一图中多条曲线--------------------"""
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x));
plt.show()

"""--------------------设置曲线颜色--------------------"""
plt.plot(x, np.sin(x - 0), color='blue') #通过颜色名称指定
plt.plot(x, np.sin(x - 1), color='g') #通过颜色简写名称指定
plt.plot(x, np.sin(x - 3), color='#FFDD44') # 6进制的RRGGBB值
#如果没有指定颜色，会在一组默认颜色值中循环使用来绘制每一条曲线。
plt.show()

"""--------------------设置曲线样式--------------------"""
# 通过具体名称设置
plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted');

# 通过形象的符号设置
plt.plot(x, x + 4, linestyle='-')  # 实线
plt.plot(x, x + 5, linestyle='--') # 虚线
plt.plot(x, x + 6, linestyle='-.') # 长短点虚线
plt.plot(x, x + 7, linestyle=':');  # 点线
plt.show()

"""--------------------调整坐标轴范围--------------------"""
plt.plot(x, np.sin(x))
plt.xlim(-1, 11) #x轴范围
plt.ylim(-1.5, 1.5); #y轴范围
plt.show()

# axis()方法
plt.plot(x, np.sin(x))
plt.axis([-1, 11, -1.5, 1.5]) #[x_min,x_max,y_min,y_max]
plt.show()

plt.plot(x, np.sin(x))
plt.axis('tight') #'tight'表示将坐标轴压缩到刚好足够绘制折线图像的大小
plt.show()

plt.plot(x, np.sin(x))
plt.axis('equal') #'equal'表示设置x轴与y轴使用相同的单位长度
plt.show()

"""--------------------标题、坐标轴标签、简单图例--------------------"""
plt.rc('text', usetex=True) #设置使用tex字体
plt.plot(x, np.sin(x))
plt.title("$y=sin(x)$") #输入tex语言的公式命令
plt.xlabel("$x$")
plt.ylabel("$sin(x)$");
plt.show()

plt.plot(x, np.sin(x), '-g', label='$sin(x)$') #这里把线条样式和颜色合并成非关键字参数
plt.plot(x, np.cos(x), ':b', label='$cos(x)$')
plt.axis('equal') #确保各个坐标轴的单位长度相等
plt.title("sin and cos")
plt.legend() #使绘制的图例线条样式和颜色与图中的曲线的风格和颜色都保持一致
plt.show()
