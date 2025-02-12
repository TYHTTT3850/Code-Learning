"""
y = k*x^2*sin(x) + 2*k + cos(x^3) k=1,2,3,4,5,6
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

x = np.linspace(0, 2*np.pi, 1000)
fig = plt.figure()
axe = fig.add_axes([0.15, 0.1, 0.8, 0.8])
for k in range(1,7):
    axe.plot(x,k*(x**2)*np.sin(x) + 2*k + np.cos(x**3),label=f"$k={k}$")
axe.legend()
plt.show()