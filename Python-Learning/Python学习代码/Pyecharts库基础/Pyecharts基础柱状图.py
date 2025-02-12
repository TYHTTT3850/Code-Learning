from pyecharts.charts import Bar
from pyecharts.options import LabelOpts # 数据标签配置项
bar = Bar() # 创建柱状图对象
bar.add_xaxis(xaxis_data=["a国","b国","c国"])
bar.add_yaxis(series_name="GDP",y_axis=[30,20,10],label_opts=LabelOpts(position="right"))
"""label_opts=LabelOpts(position="right")用于将数据标签放到柱状图右侧"""
bar.reversal_axis() # 反转x轴和y轴
bar.render("基础柱状图.html")