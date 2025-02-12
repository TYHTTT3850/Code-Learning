import numpy as np
def gauss2(A,b):
    n = len(b)
    x = np.zeros((n, 1))

    # 列主元消去
    for i in range(n):
        index = np.argmax(np.abs(A[i:, i])) + i  #选取第i列主元
        if index != i:  #交换两行
            A[i, :] = A[i, :] + A[index, :]
            A[index, :] = A[i, :] - A[index, :]
            A[i, :] = A[i, :] - A[index, :]
            b[i, :] = b[i, :] + b[index, :]
            b[index, :] = b[i, :] - b[index, :]
            b[i, :] = b[i, :] - b[index, :]

        for j in range(i + 1, n): #消元
            m = A[j, i] / A[i, i]
            A[j, :] = A[j, :] - m * A[i, :]
            b[j, :] = b[j, :] - m * b[i, :]

    # 求解
    if A[n - 1, n - 1] == 0:
        print("无解")
    else:
        x[n - 1, 0] = b[n - 1, 0] / A[n - 1, n - 1]
        for k in range(n - 2, -1, -1):
            x[k,0] = (b[k,0] - A[k,k + 1:]@x[k + 1:, 0]) / A[k,k]
    print(A)
    print(b)
    print(x)
A = np.array([[0.00000001,2,3],[-1,3.712,4.623],[-2,1.072,5.643]])
b = np.array([[1],[2],[3]],dtype=np.float64)
gauss2(A,b)
