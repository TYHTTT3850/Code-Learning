"""
创建一个时间线，每个时间点对应一张柱状图(可以改成折线图等等......)
"""
from pyecharts.charts import Bar
from pyecharts.options import LabelOpts
from pyecharts.charts import Timeline # 时间线包
from pyecharts.globals import ThemeType # 主题设置包
bar1 = Bar()
bar2 = Bar()
bar1.add_xaxis(xaxis_data=["a国","b国","c国"])
bar1.add_yaxis(series_name="GDP",y_axis=[30,20,10])
bar2.add_xaxis(xaxis_data=["a国","b国","c国"])
bar2.add_yaxis(series_name="GDP",y_axis=[50,40,30])
bar1.reversal_axis()
bar2.reversal_axis()

# 创建时间线对象并设置主题。主题设置同样适用于其他图表
timeline = Timeline({"theme":ThemeType.LIGHT})
timeline.add(bar1,"2021年GDP")
timeline.add(bar2,"2022年GDP")

# 设置自动播放
timeline.add_schema(
    play_interval=1000,# 自动播放的时间间隔，单位毫秒
    is_timeline_show=True,# 自动播放时是否显示时间线
    is_auto_play=True,# 是否自动播放
    is_loop_play=True,# 是否循环播放
)

# 使用时间线对象进行绘图
timeline.render("基础时间线柱状图.html")
