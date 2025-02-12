import numpy as np
# 高斯消元法
def gauss1(A,b):
    n = len(b)

    # 消元
    for i in range(n):
        for j in range(i+1,n):
            m = A[j,i]/A[i,i]
            A[j] = A[j] - m*A[i]
            b[j] = b[j] - m*b[i]

    # 求解
    if A[n - 1,n - 1] == 0:
        print("无解")
    else:
        x = np.zeros([n,1])
        x[n - 1] = b[n - 1] / A[n - 1,n - 1]
        for k in range(n - 2, -1, -1):
            x[k] = (b[k] -A[k,k + 1:]@x[k + 1:]) / A[k,k]
    print(A)
    print(b)
    print(x)
A = np.array([[1,0,0],[2,2,2],[4,4,5]])
b = np.array([[2],[5],[9]])
gauss1(A,b)
