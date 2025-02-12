import numpy as np
import matplotlib.pyplot as plt

def lagrange(u,X,Y): #拉格朗日插值法，u:插值点，X:已知点，Y:已知点函数值
    L = np.ones(len(X))
    for j in range(len(X)):#得到插值基函数
        for i in range(len(X)):
            if i == j:
                continue
            else:
                L[j] *= ((u-X[i])/(X[j]-X[i]))
    f = np.sum(Y*L) #累加
    return f

x = np.linspace(-5, 5, 1001)
plt.plot(x,1/(1+x**2),label = '1/(1+x^2)')

y = np.zeros(len(x))
x1 = np.linspace(-5,5,6) # 等分为5个区间
y1 = 1/(1+x1**2)
x2 = np.linspace(-5,5,11) # 等分为10个区间
y2 = 1/(1+x2**2)

for i in range(len(x)):
    y[i] = lagrange(x[i],x1,y1)
plt.plot(x,y,"--",label="L_5")

for i in range(len(x)):
    y[i] = lagrange(x[i],x2,y2)
plt.plot(x,y,"--",label="L_10")
plt.xticks(np.arange(-5,6,1))
plt.savefig("拉格朗日5次插值和10次插值.svg",format="svg")

plt.show()
