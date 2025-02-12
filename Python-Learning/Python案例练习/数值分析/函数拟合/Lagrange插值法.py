import numpy as np

def lagrange(u,X,Y): #拉格朗日插值法，u:插值点，X:已知点，Y:已知点函数值
    L = np.ones(len(X))
    for j in range(len(X)):#得到插值基函数
        for i in range(len(X)):
            if i == j:
                continue
            else:
                L[j] *= (u-X[i])/(X[j]-X[i])
    f = np.sum(Y*L)
    return f

x = np.array([144,169,225])
y = np.array([12,13,15])
print(lagrange(175,x,y))
