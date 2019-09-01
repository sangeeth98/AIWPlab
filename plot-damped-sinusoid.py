import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x = np.linspace(-1,1,1000)
pi = np.pi
plt.plot(x,2*np.exp(-3*x)*np.sin(20*pi*x),label = "exponentially decreasing")
plt.plot(x,2*np.exp(3*x)*np.sin(20*pi*x),label = "exponentially increasing")

plt.xlabel("x-axis")
plt.ylabel("y-axix")

plt.title("Sample plot")

plt.legend()

plt.show()
