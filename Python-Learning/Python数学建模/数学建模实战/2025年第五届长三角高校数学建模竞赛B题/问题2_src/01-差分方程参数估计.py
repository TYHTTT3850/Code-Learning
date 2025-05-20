import pandas as pd
import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt

# 1. 读取 Excel
df1 = pd.read_excel('../地点1热泵能耗与温差.xlsx')
Q1 = df1['热泵功率(kw)'].values
T_indoor1 = df1['平均温度'].values
T_outdoor1 = df1['环境温度(℃)'].values

df2 = pd.read_excel('../地点2热泵能耗与温差.xlsx')
Q2 = df2['热泵功率(kw)'].values
T_indoor2 = df2['平均温度'].values
T_outdoor2 = df2['环境温度(℃)'].values

# 2. 构造 X 和 y
X1 = np.column_stack((Q1[:-1], T_indoor1[:-1], T_outdoor1[:-1]))
y1 = T_indoor1[1:]  # T_in(t+1)

X2 = np.column_stack((Q2[:-1], T_indoor2[:-1], T_outdoor2[:-1]))
y2 = T_indoor2[1:]  # T_in(t+1)

# 3. 最小二乘估计参数
theta1, residuals1, rank1, s1 = lstsq(X1, y1, rcond=None)
c1, a1, b1 = theta1

theta2, residuals2, rank2, s2 = lstsq(X2, y2, rcond=None)
c2, a2, b2 = theta2

# 4. 输出结果
print(f"地点1估计结果：\nc = {c1}\na = {a1}\nb = {b1}")
print(f"地点2估计结果：\nc = {c2}\na = {a2}\nb = {b2}")

# 5. 绘图比较
fig1 = plt.figure(figsize=(16,8))
ax1 = fig1.add_subplot(1, 1, 1)
y_prediction1 = X1 @ theta1
ax1.plot(df1['采集时刻'].drop(0), y_prediction1,alpha=0.5,color='purple',label='prediction')
ax1.plot(df1['采集时刻'].drop(0),y1,"--",color='blue',label='exact')
ax1.set_xlabel('date')
ax1.set_ylabel('indoor temperature')
ax1.legend()
fig1.savefig('../问题2/地点1差分方程拟合结果.pdf',format="pdf")

fig2 = plt.figure(figsize=(16,8))
ax2 = fig2.add_subplot(1, 1, 1)
y_prediction2 = X2 @ theta2
ax2.plot(df2['采集时刻'].drop(0), y_prediction2,alpha=0.6,color='purple',label='prediction')
ax2.plot(df2['采集时刻'].drop(0),y2,"--",color='blue',label='exact')
ax2.set_xlabel('date')
ax2.set_ylabel('indoor temperature')
ax2.legend()
fig2.savefig('../问题2/地点2差分方程拟合结果.pdf',format='pdf')