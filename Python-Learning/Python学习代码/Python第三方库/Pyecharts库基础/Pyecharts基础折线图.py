# Pyecharts画折线图
from pyecharts.charts import Line
from pyecharts import options as opts
from pyecharts.options import TitleOpts # 标题配置项
from pyecharts.options import LegendOpts # 图例配置项
from pyecharts.options import ToolboxOpts # 工具箱配置项
from pyecharts.options import VisualMapOpts # 视觉映射配置项
from pyecharts.options import TooltipOpts # 提示框配置项
line = Line() # 创建折线图对象
line.add_xaxis(xaxis_data=["a国","b国","c国"])
line.add_yaxis(series_name="GDP",y_axis=[30,20,10])

# 全局配置项，set_global_opts方法。图片对象名.set_global_opts()
# 常用配置
line.set_global_opts(
    title_opts=TitleOpts(title="国家GDP展示",pos_left="center",pos_bottom="1%"),#标题配置
    legend_opts=LegendOpts(is_show=True),#图例配置
    toolbox_opts=ToolboxOpts(is_show=True),#工具箱配置
    visualmap_opts=VisualMapOpts(is_show=True),#视觉映射配置
)
line.render("Pyecharts基础折线图.html")

