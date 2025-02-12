import json
from pyecharts.charts import Map
from pyecharts.options import TitleOpts
from pyecharts.options import ToolboxOpts
from pyecharts.options import VisualMapOpts
from pyecharts import options as opts

# 读取疫情数据文件
f = open("疫情.txt","r",encoding="utf-8")
data = f.read()
f.close()

# json数据格式转换为Python数据格式
data_dict = json.loads(data)

# 取出所有省份的数据
all_province_data_list = data_dict["areaTree"][0]["children"]

# 准备画图所需的列表
data_list = []

# 取单个省份的数据
for province_data in all_province_data_list:
    province_name = province_data["name"]
    province_confirm = province_data["total"]["confirm"]
    data_list.append((province_name,province_confirm))

# 创建地图对象
map = Map(init_opts=opts.InitOpts(width="1200px", height="600px"))

# 添加地图元素
map.add("各省份确诊人数",data_list,"china")

# 设置全局选项
map.set_global_opts(
    title_opts=TitleOpts(title="全国疫情地图",pos_left="center",pos_bottom="1%"),
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
map.render("全国疫情地图.html")