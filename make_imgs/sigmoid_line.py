import numpy as np
import matplotlib.pyplot as plt
import math

x = np.arange(-1, 1, 0.1)
y = []
for x1 in x:
    y1 = (3 * x1 * x1 - 2 * x1)
    y.append(y1)
plt.plot(x, y, label = "123")
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(0, 1)
plt.legend()
plt.show()