import numpy as np
def Cholesky(A,b):
    n = len(b)
    L = np.zeros([n,n])

    # 求L
    for r in range(0,n):
        L[r,r] = np.sqrt(A[r,r]-L[r,:r]@L[r,:r].T) # 处理主对角线上的元素
        for j in range(r+1,n):
            L[j,r] = (A[j,r] - L[j,:r]@L[r,:r].T)/L[r,r]

    # 求Ly = b
    y = np.zeros([n,1])
    y[0,0] = b[0,0]/L[0,0]
    for i in range(1,n):
        y[i,0] = (b[i,0] - L[i,:i]@y[:i,0])/L[i, i]

    # 求(L^T)x = y
    L_T = L.T
    x = np.zeros([n,1])
    x[n - 1] = y[n - 1] / L_T[n - 1, n - 1]
    for k in range(n - 2, -1, -1):
        x[k,0] = (y[k,0] - L_T[k,k + 1:]@x[k + 1:,0])/L_T[k,k]
    print(L)
    print(x)

A = np.array([[1,-1,4,-2,3],
              [-1,5,0,8,-1],
              [4,0,45,8,24],
              [-2,8,8,18,4],
              [3,-1,24,4,39]],dtype=np.float64)

b = np.array([[10],
              [-2],
              [68],
              [8],
              [114]],dtype=np.float64)

Cholesky(A,b)
