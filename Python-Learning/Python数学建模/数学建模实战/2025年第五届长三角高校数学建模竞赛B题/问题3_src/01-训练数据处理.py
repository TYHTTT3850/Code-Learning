import pandas as pd
df_place1_1 = pd.read_excel("../地点1热泵能耗与温差.xlsx")
df_place1_2 = pd.read_excel("../地点1供热历史数据.xlsx")
df_place2_1 = pd.read_excel("../地点2热泵能耗与温差.xlsx")
df_place2_2 = pd.read_excel("../地点2供热历史数据.xlsx")

df_place1_1.set_index('采集时刻', inplace=True)
df_place1_2.set_index('采集时刻', inplace=True)
df_place2_1.set_index('采集时刻', inplace=True)
df_place2_2.set_index('采集时刻', inplace=True)

df_place1_ML = pd.concat([
    df_place1_1[['平均温度',"环境温度(℃)","热泵功率(kw)"]],
    df_place1_2[["供温(℃)","回温(℃)","补水流速(m3h)","设定温度(℃)"]]
], axis=1, join='inner')  # inner 保证只保留两个表中都出现的时间点

df_place2_ML = pd.concat([
    df_place2_1[['平均温度',"环境温度(℃)","热泵功率(kw)"]],
    df_place2_2[["供温(℃)","回温(℃)","补水流速(m3h)","设定温度(℃)"]]
], axis=1, join='inner')  # inner 保证只保留两个表中都出现的时间点

df_place1_ML.dropna(inplace=True)
df_place2_ML.dropna(inplace=True)

# 重置索引
df_place1_ML.reset_index(inplace=True)
df_place2_ML.reset_index(inplace=True)

df_place1_ML = df_place1_ML.rename(columns={'平均温度': '室内温度(℃)'})
df_place2_ML = df_place2_ML.rename(columns={'平均温度': '室内温度(℃)'})


# 时间特征
df_place1_ML["hour"] = df_place1_ML["采集时刻"].dt.hour  # 小时
df_place1_ML["weekday"] = df_place1_ML["采集时刻"].dt.weekday  # 星期几（0=周一）
df_place2_ML["hour"] = df_place2_ML["采集时刻"].dt.hour  # 小时
df_place2_ML["weekday"] = df_place2_ML["采集时刻"].dt.weekday  # 星期几（0=周一）

# 派生特征
df_place1_ML["供回温差"] = df_place1_ML["供温(℃)"] - df_place1_ML["回温(℃)"]
df_place1_ML["热泵是否开启"] = (df_place1_ML["热泵功率(kw)"] > 0).astype(int)
df_place1_ML["室内温度变化"] = df_place1_ML["室内温度(℃)"].diff().fillna(0)
df_place2_ML["供回温差"] = df_place2_ML["供温(℃)"] - df_place2_ML["回温(℃)"]
df_place2_ML["热泵是否开启"] = (df_place2_ML["热泵功率(kw)"] > 0).astype(int)
df_place2_ML["室内温度变化"] = df_place2_ML["室内温度(℃)"].diff().fillna(0)

# 调换列的顺序
cols = df_place1_ML.columns.tolist()
cols = [cols[0]] + cols[3:] + [cols[2]] + [cols[1]]
df_place1_ML = df_place1_ML[cols]

cols = df_place2_ML.columns.tolist()
cols = [cols[0]] + cols[3:] + [cols[2]] + [cols[1]]
df_place2_ML = df_place2_ML[cols]

df_place1_ML.to_excel("../地点1训练数据.xlsx",index=False)
df_place2_ML.to_excel("../地点2训练数据.xlsx",index=False)
