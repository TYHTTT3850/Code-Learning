from pyecharts.charts import Map
from pyecharts.options import VisualMapOpts
from pyecharts import options as opts
map = Map(init_opts=opts.InitOpts(width="1200px", height="600px"))
data = [
        ("北京市",99),
        ("上海市",199),
        ("湖南省",299),
        ("台湾省",399),
        ("广东省",499)
       ]

# 设置图名，添加数据，地图类型为中国地图
map.add("测试地图",data,"china")

#全局设置
map.set_global_opts(
    visualmap_opts=VisualMapOpts(is_show=True,
                                 is_piecewise=True,
                                 pieces=[
                                         {"min":1,"max":9,"lable":"1-9人","color":"#CCFFFF"},
                                         {"min":10,"max":99,"lable":"10-99人","color":"#FFFF99"},
                                         {"min":100,"max":499,"lable":"100-499人","color":"#FF9966"},
                                         {"min":500,"max":999,"lable":"500-999人","color":"#FF6666"},
                                         {"min":1000,"max":9999,"lable":"1000-9999人","color":"#CC3333"},
                                         {"min":10000,"lable":"10000人以上","color":"#990033"},
                                        ]
                                )
                    )
# is_piecewise表示是否手动调整视觉映射范围

# 渲染
map.render("Pyecharts基础地图.html")