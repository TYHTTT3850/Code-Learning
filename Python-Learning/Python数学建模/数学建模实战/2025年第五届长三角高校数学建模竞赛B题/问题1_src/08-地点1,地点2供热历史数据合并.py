import pandas as pd

df_1_1 = pd.read_excel("../data/地点1/供热历史数据/地点1_2022-11-15.xlsx")
df_1_2 = pd.read_excel("../data/地点1/供热历史数据/地点1_2023-11-15.xlsx")
df_1_3 = pd.read_excel("../data/地点1/供热历史数据/地点1_2024-11-15.xlsx")

df_2_1 = pd.read_excel("../data/地点2/供热历史数据/地点2_2022-11-15.xlsx")
df_2_2 = pd.read_excel("../data/地点2/供热历史数据/地点2_2023-11-15.xlsx")
df_2_3 = pd.read_excel("../data/地点2/供热历史数据/地点2_2024-11-15.xlsx")

df_place1 = pd.concat([df_1_1,df_1_2,df_1_3])
df_place2 = pd.concat([df_2_1,df_2_2,df_2_3])

df_place1.rename(columns={'时间': '采集时刻'}, inplace=True)
df_place2.rename(columns={'时间': '采集时刻'}, inplace=True)

# 转换为时间数据
df_place1['采集时刻'] = pd.to_datetime(df_place1['采集时刻'])
df_place1.sort_values('采集时刻', inplace=True)
df_place1.reset_index(drop=True, inplace=True)

df_place2['采集时刻'] = pd.to_datetime(df_place2['采集时刻'])
df_place2.sort_values('采集时刻', inplace=True)
df_place2.reset_index(drop=True, inplace=True)

df_place1.to_excel("../地点1供热历史数据.xlsx",index=False)
df_place2.to_excel("../地点2供热历史数据.xlsx",index=False)

with pd.ExcelWriter("../地点1供热历史数据.xlsx", engine='openpyxl', datetime_format='yyyy-mm-dd hh:mm:ss') as writer:
    df_place1.to_excel(writer, index=False)

with pd.ExcelWriter("../地点2供热历史数据.xlsx", engine='openpyxl', datetime_format='yyyy-mm-dd hh:mm:ss') as writer:
    df_place2.to_excel(writer, index=False)