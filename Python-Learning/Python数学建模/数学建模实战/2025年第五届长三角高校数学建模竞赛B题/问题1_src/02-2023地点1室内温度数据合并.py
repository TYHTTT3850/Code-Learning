import pandas as pd
from functools import reduce

df1 = pd.read_excel("../data/地点1/室内温度采集数据/2023/采集点1_2023111500-0072110329.xlsx")
df2 = pd.read_excel("../data/地点1/室内温度采集数据/2023/采集点2_2023111500-0072110419.xlsx")
df3 = pd.read_excel("../data/地点1/室内温度采集数据/2023/采集点3_2023111500-0072110523.xlsx")
df4 = pd.read_excel("../data/地点1/室内温度采集数据/2023/采集点4_2023111500-0072310007.xlsx")
df5 = pd.read_excel("../data/地点1/室内温度采集数据/2023/采集点5_2023111500-0072310009.xlsx")
df6 = pd.read_excel("../data/地点1/室内温度采集数据/2023/采集点6_2023111500-0072310061.xlsx")
df7 = pd.read_excel("../data/地点1/室内温度采集数据/2023/采集点7_2023111500-0072310417.xlsx")
df8 = pd.read_excel("../data/地点1/室内温度采集数据/2023/采集点8_2023111500-869324057172789.xlsx")
df9 = pd.read_excel("../data/地点1/室内温度采集数据/2023/采集点9_2023111500-869324057176459.xlsx")
df10 = pd.read_excel("../data/地点1/室内温度采集数据/2023/采集点10_2023111500-869324057186292.xlsx")

dfs = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]

for i in range(10):
    dfs[i] = dfs[i].iloc[:, 1:]  # 去掉第一列设备编号
    dfs[i].columns = ['采集时刻', f'测点{i + 1}温度']  # 重命名列

# 合并所有数据框：以“采集时刻”为主键做外连接
merged_df = reduce(lambda left,right:pd.merge(left, right, on='采集时刻', how='outer'),dfs)
# 按采集时刻排序
merged_df.sort_values('采集时刻', inplace=True)
merged_df.reset_index(drop=True, inplace=True)
merged_df.to_excel("../问题1_src/2023地点1合并结果.xlsx", index=False)