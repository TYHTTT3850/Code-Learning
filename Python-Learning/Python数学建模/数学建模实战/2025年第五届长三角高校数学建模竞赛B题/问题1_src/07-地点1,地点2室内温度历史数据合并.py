import pandas as pd

df_2022_1 = pd.read_excel("../问题1_src/2022地点1合并结果.xlsx")
df_2023_1 = pd.read_excel("../问题1_src/2023地点1合并结果.xlsx")
df_2024_1 = pd.read_excel("../问题1_src/2024地点1合并结果.xlsx")

df_2022_2 = pd.read_excel("../问题1_src/2022地点2合并结果.xlsx")
df_2023_2 = pd.read_excel("../问题1_src/2023地点2合并结果.xlsx")
df_2024_2 = pd.read_excel("../问题1_src/2024地点2合并结果.xlsx")

dfs = [df_2022_1,df_2023_1,df_2024_1,df_2022_2,df_2023_2,df_2024_2]

# 对所有测点所测温度求平均作为室内温度值
for df in dfs:
    temperature_cols = [col for col in df.columns if col.startswith('测点')]
    df['平均温度'] = df[temperature_cols].mean(axis=1, skipna=True)

df_place1 = pd.concat([df_2022_1,df_2023_1,df_2024_1])
df_place2 = pd.concat([df_2022_2,df_2023_2,df_2024_2])

# 转换时间格式
df_place1['采集时刻'] = pd.to_datetime(df_place1['采集时刻'])
df_place1.sort_values('采集时刻', inplace=True)
df_place1.reset_index(drop=True, inplace=True)

# 转换时间格式
df_place2['采集时刻'] = pd.to_datetime(df_place2['采集时刻'])
df_place2.sort_values('采集时刻', inplace=True)
df_place2.reset_index(drop=True, inplace=True)

df_place1.to_excel("../地点1室内温度历史数据.xlsx",index=False)
df_place2.to_excel("../地点2室内温度历史数据.xlsx",index=False)

with pd.ExcelWriter("../地点1室内温度历史数据.xlsx", engine='openpyxl', datetime_format='yyyy-mm-dd hh:mm:ss') as writer:
    df_place1.to_excel(writer, index=False)

with pd.ExcelWriter("../地点2室内温度历史数据.xlsx", engine='openpyxl', datetime_format='yyyy-mm-dd hh:mm:ss') as writer:
    df_place2.to_excel(writer, index=False)