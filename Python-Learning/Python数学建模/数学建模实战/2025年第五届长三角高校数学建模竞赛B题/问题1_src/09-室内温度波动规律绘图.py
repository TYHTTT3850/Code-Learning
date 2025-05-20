import pandas as pd
import matplotlib.pyplot as plt

df_place1 = pd.read_excel("../地点1室内温度历史数据.xlsx")
df_place2 = pd.read_excel("../地点2室内温度历史数据.xlsx")

# 按天取平均画图
daily_avg1 = df_place1.set_index('采集时刻')['平均温度'].resample('D').mean().reset_index()
daily_avg2 = df_place2.set_index('采集时刻')['平均温度'].resample('D').mean().reset_index()

# 地点1
fig1 = plt.figure(figsize=(16,6))
ax1_1 = fig1.add_subplot(1, 1, 1)
ax1_1.plot(daily_avg1['采集时刻'], daily_avg1['平均温度'])
fig1.tight_layout()
fig1.savefig("../问题1/地点1室内温度波动规律.pdf",format="pdf")

# 地点2
fig2 = plt.figure(figsize=(16,6))
ax2_1 = fig2.add_subplot(111)
ax2_1.plot(daily_avg2['采集时刻'], daily_avg2['平均温度'])
fig2.tight_layout()
fig2.savefig("../问题1/地点2室内温度波动规律.pdf",format="pdf")