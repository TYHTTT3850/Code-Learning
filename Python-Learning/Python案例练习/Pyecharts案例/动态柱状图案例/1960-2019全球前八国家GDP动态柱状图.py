# 作1960-2019全球前8的国家的GDP
from pyecharts.charts import Bar
from pyecharts.charts import Timeline
from pyecharts.globals import ThemeType
from pyecharts.options import *

# 排序依据函数
def Choose_Sort_Value(element):
    return element[1]

# 读取数据
file = open("1960-2019全球GDP数据.csv","r",encoding="utf-8")
data_lines = file.readlines() # 一行一行读取
del data_lines[0] # 删除第一行无用数据

# 将数据转换为字典：格式为：{年份1:[[国家1,GDP_1],[国家2,GDP_2],...],年份2:[[国家1,GDP_1],[国家2,GDP_2],...],...}
data_dict = {}
for line in data_lines:
    year = line.split(",")[0] # 获取年份
    country = line.split(",")[1] # 获取国家
    GDP = float(line.split(",")[2].replace("\n","")) # 获取GDP并将所有数据转换为浮点型
    try:
        data_dict[year].append([country,GDP])
    except KeyError:
        data_dict[year] = []
        data_dict[year].append([country,GDP])

# 先对字典的键(年份)升序排序，而不是直接for循环字典，应为直接for循环字典可能导致字典的键(年份)乱序。
sorted_keys = sorted(data_dict.keys())
timeline = Timeline({"theme":ThemeType.LIGHT})

# 构建每个时间点的柱状图
for key in sorted_keys:
    bar = Bar()
    x = []
    y = []
    data_dict[key].sort(key=Choose_Sort_Value,reverse=True)
    for country in data_dict[key][0:8]:
        x.append(country[0])
        y.append(country[1]/100000000)
    bar.add_xaxis(xaxis_data=x[::-1])
    bar.add_yaxis(series_name="全球前8国家GDP(亿)",y_axis=y[::-1],label_opts=LabelOpts(position="right"))
    bar.reversal_axis()
    bar.set_global_opts(
        title_opts=TitleOpts(title=key+"年全球前8国家GDP", pos_left="center", pos_top="5%")
    )
    timeline.add(bar,key+"年")

# 设置自动播放
timeline.add_schema(
    play_interval=1000,# 自动播放的时间间隔，单位毫秒
    is_timeline_show=True,# 自动播放时是否显示时间线
    is_auto_play=True,# 是否自动播放
    is_loop_play=True,# 是否循环播放
)
timeline.render("1960-2019全球前8国家GDP.html")
