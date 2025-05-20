import pandas as pd

df_place1_indoor = pd.read_excel("../地点1室内温度历史数据.xlsx")
df_place2_indoor = pd.read_excel("../地点2室内温度历史数据.xlsx")
df_place1_supply = pd.read_excel("../地点1供热历史数据.xlsx")
df_place2_supply = pd.read_excel("../地点2供热历史数据.xlsx")

# 设置索引为采集时刻，确保时间戳对齐
df_place1_indoor.set_index('采集时刻', inplace=True)
df_place1_supply.set_index('采集时刻', inplace=True)
df_place2_indoor.set_index('采集时刻', inplace=True)
df_place2_supply.set_index('采集时刻', inplace=True)

# 合并三列，按时间戳对齐（inner：只保留三列同时有数据的时间点）
df_merged1 = pd.concat([
    df_place1_indoor[['平均温度']],# 室内温度
    df_place1_supply[['环境温度(℃)','热泵功率(kw)']]# 室外温度
], axis=1, join='inner')  # inner 保证只保留两个表中都出现的时间点

df_merged2 = pd.concat([
    df_place2_indoor[['平均温度']],# 室内温度
    df_place2_supply[['环境温度(℃)','热泵功率(kw)']]# 室外温度
], axis=1, join='inner')  # inner 保证只保留两个表中都出现的时间点

df_merged1.dropna(inplace=True)
df_merged2.dropna(inplace=True)

df_merged1['室温与外温之差'] = df_merged1['平均温度'] - df_merged1['环境温度(℃)']
df_merged2['室温与外温之差'] = df_merged2['平均温度'] - df_merged2['环境温度(℃)']

# 重置索引
df_merged1.reset_index(inplace=True)
df_merged2.reset_index(inplace=True)

df_merged1.to_excel("../地点1热泵能耗与温差.xlsx",index=False)
df_merged2.to_excel("../地点2热泵能耗与温差.xlsx",index=False)

