# 使用pyecharts库生成2020年新冠疫情确证数据折线图
import json
from pyecharts.charts import Line
from pyecharts import options as opts


us_file = open("美国.txt","r",encoding="utf-8")
jp_file = open("日本.txt","r",encoding="utf-8")
in_file = open("印度.txt","r",encoding="utf-8")
us_data = us_file.read()
jp_data = jp_file.read()
in_data = in_file.read()

# 删除不符合json数据格式规范的字符串
us_data = us_data.replace("jsonp_1629344292311_69436(","") # 美国
us_data = us_data[:-2]

jp_data = jp_data.replace("jsonp_1629350871167_29498(","") # 日本
jp_data = jp_data[:-2]

in_data = in_data.replace("jsonp_1629350745930_63180(","") # 印度
in_data = in_data[:-2]

# json数据格式转换为python数据格式
us_data = json.loads(us_data)
jp_data = json.loads(jp_data)
in_data = json.loads(in_data)

# 获取2020年日期数据作为x轴
x_data = us_data["data"][0]["trend"]["updateDate"][:314] # 日期都一样的，所以取一份就行

# 获取2020年确诊数据
us_y_data = us_data["data"][0]["trend"]["list"][0]["data"][:314]
jp_y_data = jp_data["data"][0]["trend"]["list"][0]["data"][:314]
in_y_data = in_data["data"][0]["trend"]["list"][0]["data"][:314]

# 创建折线图对象并设置画布大小
line = Line(init_opts=opts.InitOpts(width="1200px", height="600px"))
# 添加图表数据
line.add_xaxis(x_data)
line.add_yaxis("美国确诊人数",us_y_data,label_opts=opts.LabelOpts(is_show=False))
line.add_yaxis("日本确诊人数",jp_y_data,label_opts=opts.LabelOpts(is_show=False))
line.add_yaxis("印度确诊人数",in_y_data,label_opts=opts.LabelOpts(is_show=False))

# 设置图表全局设置
line.set_global_opts(
    title_opts=opts.TitleOpts(title="2020年美日印三国新冠确诊人数折线图",pos_left="center",pos_bottom="1%"),#标题配置
    toolbox_opts=opts.ToolboxOpts(is_show=True),
)
line.render("2020年美日印三国新冠确诊人数折线图.html")

# 关闭文件
us_file.close()
jp_file.close()
in_file.close()