import pandas as pd
import matplotlib.pyplot as plt

df_place1_supply = pd.read_excel("../地点1供热历史数据.xlsx")
df_place2_supply = pd.read_excel("../地点2供热历史数据.xlsx")
df_place1_indoor = pd.read_excel("../地点1室内温度历史数据.xlsx")
df_place2_indoor = pd.read_excel("../地点2室内温度历史数据.xlsx")

# 设置时间为索引
df_place1_supply.set_index('采集时刻', inplace=True)
df_place1_indoor.set_index('采集时刻', inplace=True)
df_place2_supply.set_index('采集时刻', inplace=True)
df_place2_indoor.set_index('采集时刻', inplace=True)

# 按照天重采样
indoor_daily_1 = df_place1_indoor['平均温度'].resample('D').mean()
outdoor_daily_1 = df_place1_supply['环境温度(℃)'].resample('D').mean()
indoor_daily_2 = df_place2_indoor['平均温度'].resample('D').mean()
outdoor_daily_2 = df_place2_supply['环境温度(℃)'].resample('D').mean()

df_merge_1 = pd.concat([indoor_daily_1, outdoor_daily_1], axis=1)
df_merge_2 = pd.concat([indoor_daily_2, outdoor_daily_2], axis=1)
df_merge_1.columns = ['室内温度', '环境温度']
df_merge_2.columns = ['室内温度', '环境温度']

# 计算皮尔逊相关系数
corr1 = df_merge_1['室内温度'].corr(df_merge_1['环境温度']) #-0.0443
corr2 = df_merge_2['室内温度'].corr(df_merge_2['环境温度']) #0.0632
print(f"地点1 室内温度与环境温度的相关系数：{corr1:.4f}")
print(f"地点2 室内温度与环境温度的相关系数：{corr2:.4f}")

fig1 = plt.figure(figsize=(16, 8))
fig2 = plt.figure(figsize=(16, 8))
ax1 = fig1.add_subplot(1, 1, 1)
ax2 = fig2.add_subplot(1, 1, 1)
ax1.scatter(df_merge_1['室内温度'], df_merge_1['环境温度'], alpha=0.6, color='purple')
ax2.scatter(df_merge_2['室内温度'], df_merge_2['环境温度'], alpha=0.6, color='purple')
ax1.set_xlabel('indoor temperature')
ax2.set_xlabel('indoor temperature')
ax1.set_ylabel('outdoor temperature')
ax2.set_ylabel('outdoor temperature')
fig1.tight_layout()
fig2.tight_layout()
fig1.savefig("../问题1/地点1室内外相关相关性曲线.pdf",format="pdf")
fig2.savefig("../问题1/地点2室内外相关相关性曲线.pdf",format="pdf")