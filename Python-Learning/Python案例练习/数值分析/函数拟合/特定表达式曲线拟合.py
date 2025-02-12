import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func(x,c):
    return x/(c*x-2*c)

x_data = np.array([2.2500,2.5417,2.8333,3.1250,3.4167,3.7083,4.000,0.7000,
               1.0714,1.4299,1.8143,2.1857,2.5571,2.9286,3.3000])
y_data = np.array([2.8648,1.4936,1.0832,0.8842,0.7677,0.6910,0.6366,-0.1714,
               -0.3673,-0.8243,-3.1096,3.7463,1.4610,1.0039,0.8080])
plt.scatter(x_data,y_data)

popt,pcov = curve_fit(func,x_data,y_data)
print(popt)
x1 = np.linspace(0.6,1.9,1301)
y1 = func(x1,*popt)
x2 = np.linspace(2.1,4.1,2001)
y2 = func(x2,*popt)
plt.plot(x1,y1,'r--')
plt.plot(x2,y2,'r--')
plt.ylim([-10,10])
plt.show()
