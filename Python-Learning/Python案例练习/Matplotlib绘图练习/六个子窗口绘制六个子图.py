"""
y = k*x^2*sin(x) + 2*k + cos(x^3) k=1,2,3,4,5,6
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

fig,axes = plt.subplots(2,3)
x = np.linspace(0, 2*np.pi, 1000)
for i in range(2):
    for j in range(3):
        k = i*2+j+1
        axes[i,j].plot(x,k*(x**2)*np.sin(x) + 2*k + np.cos(x**3),label=f"$k={k}$")
        axes[i,j].legend()

plt.tight_layout()
plt.show()
