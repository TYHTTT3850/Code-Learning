#鞍点：矩阵改位置的元素是该行的最大值，该列的最小值。矩阵可能存在对各鞍点，也有可能没有鞍点
import numpy as np
B = []
A = np.array([[9,80,205,40],[90,-60,96,1],[210,-3,101,89]])
print(A)
rows,columns = A.shape
print(f"行：{rows},列：{columns}")
for i in range(rows):
    max = A[i][0]
    for j in range(0,columns):
        if A[i][j] > max:
            max = A[i][j]
    B.append(max)
print("每行的最大值为：\n",B)
for i in range(rows):
    x,y = np.where(A == B[i])
    min = A[0,y[0]]
    for j in range(0,rows):
        if A[j,y[0]] <= min:
            min = A[j,y[0]]
    if min == B[i]:
        print(f"{B[i]}为鞍点")