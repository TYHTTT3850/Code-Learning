import pandas as pd
import matplotlib.pyplot as plt

df_power_temperature1 = pd.read_excel("../地点1热泵能耗与温差.xlsx")
df_power_temperature2 = pd.read_excel("../地点2热泵能耗与温差.xlsx")

# 按照天取平均
df_power_temperature1.set_index('采集时刻',inplace=True)
df_power_temperature2.set_index('采集时刻',inplace=True)
daily_avg1 = df_power_temperature1.resample('D').mean()
daily_avg2 = df_power_temperature2.resample('D').mean()

# 计算皮尔逊相关系数
corr1 = df_power_temperature1['热泵功率(kw)'].corr(df_power_temperature1['室温与外温之差'])
corr2 = df_power_temperature2['热泵功率(kw)'].corr(df_power_temperature2['室温与外温之差'])
print(f"地点1 热泵功率与温度差相关系数：{corr1:.4f}") #0.1366
print(f"地点1 热泵功率与温度差相关系数：{corr2:.4f}") #-0.0897

fig1 = plt.figure(figsize=(16,8))
fig2 = plt.figure(figsize=(16,8))
ax1 = fig1.add_subplot(1, 1, 1)
ax2 = fig2.add_subplot(1, 1, 1)

ax1.scatter(daily_avg1['室温与外温之差'],daily_avg1['热泵功率(kw)'], alpha=0.6, color='purple')
ax2.scatter(daily_avg2['室温与外温之差'],daily_avg2['热泵功率(kw)'], alpha=0.6, color='purple')

ax1.set_xlabel('temperature difference')
ax2.set_xlabel('temperature difference')
ax1.set_ylabel('power')
ax2.set_ylabel('power')

fig1.tight_layout()
fig2.tight_layout()

fig1.savefig("../问题1/地点1热泵能耗与温度差相关性散点.pdf",format='pdf')
fig2.savefig("../问题1/地点2热泵能耗与温度差相关性散点.pdf",format='pdf')