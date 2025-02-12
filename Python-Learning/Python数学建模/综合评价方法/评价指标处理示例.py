import numpy as np
import pandas as pd

a = np.loadtxt("性能指标数据.txt", dtype=str, encoding='utf-8')
a = a[1:,:].astype(float)
R1 = a.copy()
R2 = a.copy()
R3 = a.copy()

for j in [0,1,2,4,5]: #极大化指标处理
    R1[:,j] = R1[:,j] / np.sqrt(np.sum(R1[:,j]**2)) #向量归一化
    R2[:,j] = R2[:,j] / max(R2[:,j]) #比例变换法
    R3[:,j] = (R3[:,j]-min(R3[:,j])) / (max(R3[:,j])-min(R3[:,j])) #极差变换法
# 极小化指标处理
R1[:,3] = R1[:,3] / np.sqrt(np.sum(R1[:,3]**2))
R2[:,3] = 1 - R2[:,3] / max(R2[:,3])
R3[:,3] = (max(R3[:,3])-R3[:,3]) / (max(R3[:,3])-min(R3[:,3]))

print(R1,R2,R3,sep='\n\n')