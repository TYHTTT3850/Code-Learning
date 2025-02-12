# 作1960-2019全球前8的国家的GDP
import pandas as pd
from pyecharts.charts import Bar
from pyecharts.charts import Timeline
from pyecharts.globals import ThemeType
from pyecharts.options import *

# 使用Pandas库读取数据
df:pd.DataFrame = pd.read_csv("1960-2019全球GDP数据.csv", index_col=["year","country"],
                                dtype={"year":str, "country":str, "gdp":int}) #读取后会自动排序

# 获取所有年份，存在列表中
years = list(set(df.index.get_level_values(0).tolist())) #列表转换成集合去掉重复后再换为列表
years.sort() #按照大小排序

# 时间线作图初始化
timeline = Timeline({"theme":ThemeType.LIGHT})


# 遍历所有年份，处理数据
for year in years:
    country = df.loc[year].head(8).index.tolist() #得到每年前八国家的名称，封装在列表中
    GDP = df.loc[year].head(8).values.tolist() #得到每年前八国家的GDP，封装在列表中
    for i in range(8):
        GDP[i] = GDP[i][0]/100000000
    bar = Bar()
    bar.add_xaxis(xaxis_data=country[::-1])
    bar.add_yaxis(series_name="全球前8国家GDP(亿)", y_axis=GDP[::-1], label_opts=LabelOpts(position="right"))
    bar.reversal_axis()
    bar.set_global_opts(
        title_opts=TitleOpts(title=year + "年全球前8国家GDP", pos_left="center", pos_top="5%")
    )
    timeline.add(bar, year + "年")

# 时间线渲染设置
timeline.add_schema(
    play_interval=1000,# 自动播放的时间间隔，单位毫秒
    is_timeline_show=True,# 自动播放时是否显示时间线
    is_auto_play=True,# 是否自动播放
    is_loop_play=True,# 是否循环播放
)
timeline.render("1960-2019全球前8国家GDP.html")