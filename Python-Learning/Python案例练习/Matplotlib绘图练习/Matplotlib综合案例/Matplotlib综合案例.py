"""
给定某市场2009.1.1~2012.12.30的交易数据，对其中的一些数据特征进行可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='SimSun') #设置全局字体为宋体
fig = plt.figure(figsize=(10, 5))

a = pd.read_excel("交易数据.xlsx")
a['year'] = a.Date.dt.year #添加交易年份字段
a['month'] = a.Date.dt.month #添加交易月份字段
"""
dt是Pandas提供的一个属性访问器，用于操作包含日期时间数据的列。
通过dt.year可以提取日期数据中的年份信息，dt.month可以提取日期数据中的月份信息。
"""
axe1 = fig.add_subplot(2,3,1)
axe1.set_aspect('equal') #设置子图纵横比例相等
Class_Counts = a.Order_Class[a.year==2012].value_counts()
"""
.value_counts()：用于统计指定列中每种类别的出现次数，并返回一个包含类别和其对应计数的Series对象。
"""
Class_Percents = Class_Counts/Class_Counts.sum()
axe1.pie(Class_Percents,labels=Class_Counts.index,autopct='%.1f%%')
"""
%.1f：格式化字符串，表示保留1位小数。
%%：双百分号表示在标签中显示一个%符号。
"""
axe1.set_title("2012年各等级订单比例")

axe2 = fig.add_subplot(2,3,2)
Month_Sales = a[a.year==2012].groupby(by='month').aggregate({'Sales': 'sum'})
"""
.groupby(by='month')：对筛选后的数据框按month列进行分组。即相同月份的数据会被放在同一组中。
.aggregate({'Sales': np.sum})：对每个月的Sales列进行聚合计算。这里使用sum求和计算每个月的总销售额。
"""
axe2.plot(np.arange(1, 13), Month_Sales['Sales'])
axe2.set_title("2012年各月销售趋势")

axe3 = fig.add_subplot(2,3,(3,6))
cost = a['Trans_Cost'].groupby(by = a['Transport'])
"""
GroupBy对象cost并不直接包含Trans_Cost的具体数据，它是一个按Transport列分组的“视图”，知道每个分组对应的索引和位置
"""
ts = list(cost.groups.keys())
d1 = cost.get_group('大卡')
"""
通过get_group()方法，可以根据从cost中的分组规则访问到原始数据框a中相应的Trans_Cost列数据。
"""
d2 = cost.get_group('火车')
d3 = cost.get_group('空运')
axe3.boxplot([d1, d2, d3])
axe3.set_xticklabels(ts)

axe4 = fig.add_subplot(2,3,(4,5))
axe4.hist(a.Sales[a.year==2012],bins=40,density=True)
axe4.set_title("2012年销售额分布图")
axe4.set_xlabel("销售额")

fig.tight_layout()
plt.show()
