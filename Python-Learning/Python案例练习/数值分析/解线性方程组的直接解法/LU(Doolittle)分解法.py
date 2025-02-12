import numpy as np
def Doolittle(A,b):
    # LU分解
    n = len(b)
    L = np.eye(n)
    U = np.zeros([n,n])
    U[0,:] = A[0,:]
    L[1:,0] = A[1:,0]/U[0,0]
    for r in range(1,n):
        for i in range(r,n):
            U[r,i] = A[r,i] - L[r,:r]@U[:r,i]
            L[i,r] = (A[i,r]-L[i,:r]@U[:r,r])/U[r,r]

    # 求解Ly = b
    y = np.zeros([n,1])
    y[0,0] = b[0,0]
    for i in range(1,n):
        y[i,0] = b[i,0] - L[i,:i]@y[:i,0]

    # 求解Ux = y
    x = np.zeros([n,1])
    x[n - 1] = y[n - 1] / U[n - 1, n - 1]
    for k in range(n - 2, -1, -1):
        x[k,0] = (y[k,0] - U[k, k + 1:]@x[k + 1:,0]) / U[k, k]
    print(L)
    print(U)
    print(x)

A = np.array([[2,10,0,-3],[-3,-4,-12,13],[1,2,3,-4],[4,14,9,-13]])
b = np.array([[10],[5],[-2],[7]])
Doolittle(A,b)
