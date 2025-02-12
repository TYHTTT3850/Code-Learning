import json
from pyecharts.charts import Map
from pyecharts.options import TitleOpts
from pyecharts.options import ToolboxOpts
from pyecharts.options import VisualMapOpts
from pyecharts import options as opts
f = open("疫情.txt","r",encoding="utf-8")
data = f.read()
data = json.loads(data)
f.close()
data_henan = data["areaTree"][0]["children"][3]["children"]
data_list = []
for city in data_henan:
    city_name = city["name"]+"市"
    city_confirm = city["total"]["confirm"]
    data_list.append((city_name,city_confirm))
map = Map(init_opts=opts.InitOpts(width="1200px", height="600px"))
map.add("河南各市确诊人数",data_list,"河南")
map.set_global_opts(
    title_opts=TitleOpts(title="河南省疫情地图",pos_left="center",pos_bottom="1%"),
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
                                ),
    toolbox_opts=ToolboxOpts(is_show=True)
                   )

map.render("河南省疫情地图.html")